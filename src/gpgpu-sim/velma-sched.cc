#include "velma_sched.h"

velma_gto_scheduler::velma_gto_scheduler(const shader_core_config* config, 
                                         const memory_config* mem_config, 
                                         shader_core_ctx* core, 
                                         memory_stats_t* stats,
                                         std::vector<shd_warp_t>* warp)
    : gto_scheduler(config, mem_config, core, stats, warp), 
      m_velma_unit(new velma(config, mem_config, core, stats)) {
}

velma_gto_scheduler::~velma_gto_scheduler() {
    delete m_velma_unit;
}

void velma_gto_scheduler::cycle() {
    if (m_velma_unit->is_busy()) {
        // If velma is busy, skip issuing new instructions to it
        return;
    }

    // Call the base class cycle method
    gto_scheduler::cycle();
}

void velma_gto_scheduler::issue(register_set& pipe_reg_set) {
    warp_inst_t* inst = pipe_reg_set.get_ready_inst();
    if (inst) {
        if (inst->is_custom_instruction()) {
            m_velma_unit->issue(inst); // Use velma for custom instructions
        } else {
            gto_scheduler::issue(pipe_reg_set); // Call the base class issue method
        }
    }
}

