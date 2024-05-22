#include "velma_sched.h"

velma_gto_scheduler::velma_gto_scheduler(const shader_core_config* config, shader_core_ctx* core, memory_stats_t* stats)
    : m_config(config), m_core(core), m_stats(stats), m_velma_unit(new velma(config, core, stats)) {
    for (unsigned i = 0; i < m_config->max_warps_per_shader; ++i) {
        m_supervised_warps.push_back(&m_core->m_warp[i]);
    }
}

velma_gto_scheduler::~velma_gto_scheduler() {
    delete m_velma_unit;
}

void velma_gto_scheduler::cycle() {
    SCHED_DPRINTF("scheduler_unit::cycle()\n");
    bool valid_inst = false;
    bool ready_inst = false;
    bool issued_inst = false;

    // Check if velma is busy, and if not, try to issue an instruction to velma
    if (!m_velma_unit->is_busy()) {
        for (auto& warp : m_supervised_warps) {
            if (!warp->done_exit()) {
                const warp_inst_t* inst = warp->ibuffer_next_inst();
                if (inst && inst->is_load()) {
                    m_velma_unit->issue(inst);
                    issued_inst = true;
                    break;
                }
            }
        }
    }

    order_warps();
    //gotta figure out how to reorder the warps. 
    for (std::vector<shd_warp_t*>::const_iterator iter = m_next_cycle_prioritized_warps.begin(); iter != m_next_cycle_prioritized_warps.end(); iter++) {
        if ((*iter) == NULL || (*iter)->done_exit()) {
            continue;
        }
        SCHED_DPRINTF("Testing (warp_id %u, dynamic_warp_id %u)\n", (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
        unsigned warp_id = (*iter)->get_warp_id();
        unsigned checked = 0;
        unsigned issued = 0;
        exec_unit_type_t previous_issued_inst_exec_type = exec_unit_type_t::NONE;
        unsigned max_issue = m_shader->m_config->gpgpu_max_insn_issue_per_warp;
        bool diff_exec_units = m_shader->m_config->gpgpu_dual_issue_diff_exec_units;

        if (warp(warp_id).ibuffer_empty())
            SCHED_DPRINTF("Warp (warp_id %u, dynamic_warp_id %u) fails as ibuffer_empty\n", (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());

        if (warp(warp_id).waiting())
            SCHED_DPRINTF("Warp (warp_id %u, dynamic_warp_id %u) fails as waiting for barrier\n", (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());

        while (!warp(warp_id).waiting() && !warp(warp_id).ibuffer_empty() && (checked < max_issue) && (checked <= issued) && (issued < max_issue)) {
            const warp_inst_t *pI = warp(warp_id).ibuffer_next_inst();
            if (pI && pI->m_is_cdp && warp(warp_id).m_cdp_latency > 0) {
                assert(warp(warp_id).m_cdp_dummy);
                warp(warp_id).m_cdp_latency--;
                break;
            }

            bool valid = warp(warp_id).ibuffer_next_valid();
            bool warp_inst_issued = false;
            unsigned pc, rpc;
            m_shader->get_pdom_stack_top_info(warp_id, pI, &pc, &rpc);
            SCHED_DPRINTF("Warp (warp_id %u, dynamic_warp_id %u) has valid instruction (%s)\n", (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(), m_shader->m_config->gpgpu_ctx->func_sim->ptx_get_insn_str(pc).c_str());
            if (pI) {
                assert(valid);
                if (pc != pI->pc) {
                    SCHED_DPRINTF("Warp (warp_id %u, dynamic_warp_id %u) control hazard instruction flush\n", (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
                    warp(warp_id).set_next_pc(pc);
                    warp(warp_id).ibuffer_flush();
                } else {
                    valid_inst = true;
                    if (!m_scoreboard->checkCollision(warp_id, pI)) {
                        SCHED_DPRINTF("Warp (warp_id %u, dynamic_warp_id %u) passes scoreboard\n", (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
                        ready_inst = true;

                        const active_mask_t &active_mask = m_shader->get_active_mask(warp_id, pI);
                        assert(warp(warp_id).inst_in_pipeline());

                        if ((pI->op == LOAD_OP) || (pI->op == STORE_OP) || (pI->op == MEMORY_BARRIER_OP) || (pI->op == TENSOR_CORE_LOAD_OP) || (pI->op == TENSOR_CORE_STORE_OP)) {
                            if (m_mem_out->has_free(m_shader->m_config->sub_core_model, m_id) && (!diff_exec_units || previous_issued_inst_exec_type != exec_unit_type_t::MEM)) {
                                m_shader->issue_warp(*m_mem_out, pI, active_mask, warp_id, m_id);
                                issued++;
                                issued_inst = true;
                                warp_inst_issued = true;
                                previous_issued_inst_exec_type = exec_unit_type_t::MEM;
                            }
                        } else {
                            bool sp_pipe_avail = (m_shader->m_config->gpgpu_num_sp_units > 0) && m_sp_out->has_free(m_shader->m_config->sub_core_model, m_id);
                            bool sfu_pipe_avail = (m_shader->m_config->gpgpu_num_sfu_units > 0) && m_sfu_out->has_free(m_shader->m_config->sub_core_model, m_id);
                            bool tensor_core_pipe_avail = (m_shader->m_config->gpgpu_num_tensor_core_units > 0) && m_tensor_core_out->has_free(m_shader->m_config->sub_core_model, m_id);
                            bool dp_pipe_avail = (m_shader->m_config->gpgpu_num_dp_units > 0) && m_dp_out->has_free(m_shader->m_config->sub_core_model, m_id);
                            bool int_pipe_avail = (m_shader->m_config->gpgpu_num_int_units > 0) && m_int_out->has_free(m_shader->m_config->sub_core_model, m_id);

                            if (pI->op != TENSOR_CORE_OP && pI->op != SFU_OP && pI->op != DP_OP && !(pI->op >= SPEC_UNIT_START_ID)) {
                                bool execute_on_SP = false;
                                bool execute_on_INT = false;

                                if (m_shader->m_config->gpgpu_num_int_units > 0 && int_pipe_avail && pI->op != SP_OP && !(diff_exec_units && previous_issued_inst_exec_type == exec_unit_type_t::INT))
                                    execute_on_INT = true;
                                else if (sp_pipe_avail && (m_shader->m_config->gpgpu_num_int_units == 0 || (m_shader->m_config->gpgpu_num_int_units > 0 && pI->op == SP_OP)) && !(diff_exec_units && previous_issued_inst_exec_type == exec_unit_type_t::SP))
                                    execute_on_SP = true;

                                if (execute_on_INT || execute_on_SP) {
                                    if (pI->m_is_cdp && !warp(warp_id).m_cdp_dummy) {
                                        assert(warp(warp_id).m_cdp_latency == 0);

                                        if (pI->m_is_cdp == 1)
                                            warp(warp_id).m_cdp_latency = m_shader->m_config->gpgpu_ctx->func_sim->cdp_latency[pI->m_is_cdp - 1];
                                        else
                                            warp(warp_id).m_cdp_latency = m_shader->m_config->gpgpu_ctx->func_sim->cdp_latency[pI->m_is_cdp - 1] + m_shader->m_config->gpgpu_ctx->func_sim->cdp_latency[pI->m_is_cdp] * active_mask.count();
                                        warp(warp_id).m_cdp_dummy = true;
                                        break;
                                    } else if (pI->m_is_cdp && warp(warp_id).m_cdp_dummy) {
                                        assert(warp(warp_id).m_cdp_latency == 0);
                                        warp(warp_id).m_cdp_dummy = false;
                                    }
                                }

                                if (execute_on_SP) {
                                    m_shader->issue_warp(*m_sp_out, pI, active_mask, warp_id, m_id);
                                    issued++;
                                    issued_inst = true;
                                    warp_inst_issued = true;
                                    previous_issued_inst_exec_type = exec_unit_type_t::SP;
                                } else if (execute_on_INT) {
                                    m_shader->issue_warp(*m_int_out, pI, active_mask, warp_id, m_id);
                                    issued++;
                                    issued_inst = true;
                                    warp_inst_issued = true;
                                    previous_issued_inst_exec_type = exec_unit_type_t::INT;
                                }
                            } else if ((m_shader->m_config->gpgpu_num_dp_units > 0) && (pI->op == DP_OP) && !(diff_exec_units && previous_issued_inst_exec_type == exec_unit_type_t::DP)) {
                                if (dp_pipe_avail) {
                                    m_shader->issue_warp(*m_dp_out, pI, active_mask, warp_id, m_id);
                                    issued++;
                                    issued_inst = true;
                                    warp_inst_issued = true;
                                    previous_issued_inst_exec_type = exec_unit_type_t::DP;
                                }
                            } else if (((m_shader->m_config->gpgpu_num_dp_units == 0 && pI->op == DP_OP) || (pI->op == SFU_OP) || (pI->op == ALU_SFU_OP)) && !(diff_exec_units && previous_issued_inst_exec_type == exec_unit_type_t::SFU)) {
                                if (sfu_pipe_avail) {
                                    m_shader->issue_warp(*m_sfu_out, pI, active_mask, warp_id, m_id);
                                    issued++;
                                    issued_inst = true;
                                    warp_inst_issued = true;
                                    previous_issued_inst_exec_type = exec_unit_type_t::SFU;
                                }
                            } else if ((pI->op == TENSOR_CORE_OP) && !(diff_exec_units && previous_issued_inst_exec_type == exec_unit_type_t::TENSOR)) {
                                if (tensor_core_pipe_avail) {
                                    m_shader->issue_warp(*m_tensor_core_out, pI, active_mask, warp_id, m_id);
                                    issued++;
                                    issued_inst = true;
                                    warp_inst_issued = true;
                                    previous_issued_inst_exec_type = exec_unit_type_t::TENSOR;
                                }
                            } else if ((pI->op >= SPEC_UNIT_START_ID) && !(diff_exec_units && previous_issued_inst_exec_type == exec_unit_type_t::SPECIALIZED)) {
                                unsigned spec_id = pI->op - SPEC_UNIT_START_ID;
                                assert(spec_id < m_shader->m_config->m_specialized_unit.size());
                                register_set *spec_reg_set = m_spec_cores_out[spec_id];
                                bool spec_pipe_avail = (m_shader->m_config->m_specialized_unit[spec_id].num_units > 0) && spec_reg_set->has_free(m_shader->m_config->sub_core_model, m_id);

                                if (spec_pipe_avail) {
                                    m_shader->issue_warp(*spec_reg_set, pI, active_mask, warp_id, m_id);
                                    issued++;
                                    issued_inst = true;
                                    warp_inst_issued = true;
                                    previous_issued_inst_exec_type = exec_unit_type_t::SPECIALIZED;
                                }
                            }

                        }
                    } else {
                        SCHED_DPRINTF("Warp (warp_id %u, dynamic_warp_id %u) fails scoreboard\n", (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
                    }
                }
            } else if (valid) {
                SCHED_DPRINTF("Warp (warp_id %u, dynamic_warp_id %u) return from diverged warp flush\n", (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
                warp(warp_id).set_next_pc(pc);
                warp(warp_id).ibuffer_flush();
            }
            if (warp_inst_issued) {
                SCHED_DPRINTF("Warp (warp_id %u, dynamic_warp_id %u) issued %u instructions\n", (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(), issued);
                do_on_warp_issued(warp_id, issued, iter);
            }
            checked++;
        }
        if (issued) {
            for (std::vector<shd_warp_t*>::const_iterator supervised_iter = m_supervised_warps.begin(); supervised_iter != m_supervised_warps.end(); ++supervised_iter) {
                if (*iter == *supervised_iter) {
                    m_last_supervised_issued = supervised_iter;
                }
            }

            if (issued == 1)
                m_stats->single_issue_nums[m_id]++;
            else if (issued > 1)
                m_stats->dual_issue_nums[m_id]++;
            else
                abort();

            break;
        }
    }

    if (!valid_inst)
        m_stats->shader_cycle_distro[0]++;
    else if (!ready_inst)
        m_stats->shader_cycle_distro[1]++;
    else if (!issued_inst)
        m_stats->shader_cycle_distro[2]++;
}

