#include "velma.h"

#include <string.h>
#include "../abstract_hardware_model.h"
#include "../../libcuda/gpgpu_context.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_sim.h"
#include "../statwrapper.h"
#include "addrdec.h"
#include "dram.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "icnt_wrapper.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader_trace.h"
#include "stat-tool.h"
#include "traffic_breakdown.h"
#include "visualizer.h"



/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   velma_entry_t   /////////////////////////
/////////////////////////////////////////////////////////////// 

velma_entry_t::velma_entry_t(velma_pc_t pc_, velma_id_t vid){
      pc = pc_; 
      velma_id = vid; 
      //initialize the warpcluster mask to all 1s! 
      wc_mask = std::bitset<VELMA_WARPCLUSTER_SIZE>((1ULL << VELMA_WARPCLUSTER_SIZE) - 1); 
      killtimer = VELMA_KILLTIMER_START;
    } 

inline void velma_entry_t::mark_warp_reached(warp_id_t wid){
  uint8_t warp_index = wid % VELMA_WARPCLUSTER_SIZE;  
  wc_mask.set(warp_index);
}

inline bool velma_entry_t::has_warp_reached(warp_id_t wid){
  uint8_t warp_index = wid % VELMA_WARPCLUSTER_SIZE;
  return static_cast<bool>(wc_mask[warp_index]);
} 

//decrements the killtimer of the entry. if it reaches 0,
//return -1 as the velma_id. 
velma_id_t velma_entry_t::decr_killtimer(){
  velma_id_t vid = -1;
  if (killtimer > 0){
    killtimer--;
    vid = velma_id;
  }
  return vid;
}


//////////////////////////////////////////////////////////////////////////////////////
////////////////////////////  warpcluster_entry_t  //////////////
/////////////////////////////////////////////////////////

velma_id_t warpcluster_entry_t::active_velma_id(){
  velma_id_t retvid = -1;
  if (!velma_entries.empty()){
    retvid = velma_entries.begin()->velma_id;
  }
  return retvid;
}

std::set<velma_id_t> warpcluster_entry_t::all_velma_ids_from_pc(velma_pc_t pc){
  std::set<velma_id_t> vids;
  for (auto velma_entry : velma_entries){
    if (velma_entry.pc == pc){
      vids.insert(velma_entry.velma_id);
    }
  }
  return vids;
}

//Checks this cluster's subtable for the pc in question.
//We only care about the first instance.
velma_id_t warpcluster_entry_t::first_velma_id_from_pc(velma_pc_t pc){
  for (auto velma_entry : velma_entries){
    if (velma_entry.pc == pc){
      return velma_entry.velma_id;
    }
  }
  return -1;
}

//checks if a vid is present in this cluster's subtable
bool warpcluster_entry_t::contains_velma_id(velma_id_t vid){
  for (auto velma_entry : velma_entries){
    if (velma_entry.velma_id == vid and vid != -1){
      return true;
    }
  }
  return false;
}


//decrements the top killtimer and returns the associated vid. 
velma_id_t warpcluster_entry_t::decr_top_killtimer(){
  velma_id_t decr_vid = -1;
  //no segfaults, please. 
  if (!velma_entries.empty()){ 
    decr_vid = velma_entries.begin()->decr_killtimer();
  }
  return decr_vid;
}

//reports the set of velma ids this cluster is tracking. 
std::set<velma_id_t> warpcluster_entry_t::report_velma_ids(){
  std::set<velma_id_t> vids;
  for (auto velma_entry : velma_entries){
    vids.insert(velma_entry.velma_id);
  }
  return vids;
}

/* Pops the top velma entry, advancing the queue.
 * also returns the velma id of that element,
 * or -1 if the list is empty.
 */
velma_id_t warpcluster_entry_t::pop_front_velma_entry(){
  velma_id_t ret_vid = -1; 
  if(!velma_entries.empty()){
    ret_vid = velma_entries.begin()->velma_id;
    velma_entries.pop_front(); 
  }
  return ret_vid; 
}


velma_id_t warpcluster_entry_t::mark_warp_reached_pc(warp_id_t wid, velma_pc_t pc){
  for (velma_entry_t& entry : velma_entries){
    //does this entry correspond to the pc we care about? 
    if (entry.pc == pc){
      //if this warp isn't marked, mark it, and return the id!
      if (!entry.has_warp_reached(wid)){
        entry.mark_warp_reached(wid);
        return entry.velma_id;
      }
    }
  }
  return -1;
}

//////////////////////////////////////////////////////////////////////////////////
/////////////////////////   SCHEDULER       /////////////////////////
///////////////////////////////////////////////////
void velma_scheduler::order_warps() {
  order_velma_lrr(m_next_cycle_prioritized_warps, m_supervised_warps,
            m_last_supervised_issued, m_supervised_warps.size());
}

void velma_scheduler::velma_cycle()
{
  //decrement our pc killtimers. Dead timers yield bodies!
  std::set<velma_id_t> expiring_velma_ids = decr_pc_killtimers();
  //erase the velma entries corresponding to those velma_ids. 
  for (velma_id_t exvid : expiring_velma_ids){
    clear_velma_entry(exvid); 
  }
  just_expired_velma_ids = expiring_velma_ids; 
}





void velma_scheduler::cycle(){
  SCHED_DPRINTF("scheduler_unit::cycle()\n");
  bool valid_inst =
      false;  // there was one warp with a valid instruction to issue (didn't
              // require flush due to control hazard)
  bool ready_inst = false;   // of the valid instructions, there was one not
                             // waiting for pending register writes
  bool issued_inst = false;  // of these we issued one
                             //
  //cycle here. this decrements the killtimers and cleans up as necessary before 
  //we order the warps. 
  velma_cycle();


  order_warps();
  for (std::vector<shd_warp_t *>::const_iterator iter =
           m_next_cycle_prioritized_warps.begin();
       iter != m_next_cycle_prioritized_warps.end(); iter++) {
    // Don't consider warps that are not yet valid
    if ((*iter) == NULL || (*iter)->done_exit()) {
      continue;
    }

    SCHED_DPRINTF("Testing (warp_id %u, dynamic_warp_id %u)\n",
                  (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
    unsigned warp_id = (*iter)->get_warp_id();
    unsigned checked = 0;
    unsigned issued = 0;
    exec_unit_type_t previous_issued_inst_exec_type = exec_unit_type_t::NONE;
    unsigned max_issue = m_shader->m_config->gpgpu_max_insn_issue_per_warp;
    bool diff_exec_units =
        m_shader->m_config
            ->gpgpu_dual_issue_diff_exec_units;  // In tis mode, we only allow
                                                 // dual issue to diff execution
                                                 // units (as in Maxwell and
                                                 // Pascal)

    if (warp(warp_id).ibuffer_empty()){
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) fails as ibuffer_empty\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
    }

    if (warp(warp_id).waiting()){
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) fails as waiting for "
          "barrier\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
    }

    while (!warp(warp_id).waiting() && !warp(warp_id).ibuffer_empty() &&
           (checked < max_issue) && (checked <= issued) &&
           (issued < max_issue)) 
    { //get the next instruction for this warp 
      const warp_inst_t *pI = warp(warp_id).ibuffer_next_inst();
      
      // Jin: handle cdp latency;
      if (pI && pI->m_is_cdp && warp(warp_id).m_cdp_latency > 0) {
        assert(warp(warp_id).m_cdp_dummy);
        warp(warp_id).m_cdp_latency--;
        break;
      }

      bool valid = warp(warp_id).ibuffer_next_valid();
      bool warp_inst_issued = false;
      unsigned pc, rpc;
      m_shader->get_pdom_stack_top_info(warp_id, pI, &pc, &rpc);
      SCHED_DPRINTF(
          "Warp (warp_id %u, dynamic_warp_id %u) has valid instruction (%s)\n",
          (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(),
          m_shader->m_config->gpgpu_ctx->func_sim->ptx_get_insn_str(pc).c_str());

      if (pI) {
        assert(valid);
        if (pc != pI->pc) 
        {
          SCHED_DPRINTF(
              "Warp (warp_id %u, dynamic_warp_id %u) control hazard "
              "instruction flush\n",
              (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
          // control hazard
          warp(warp_id).set_next_pc(pc);
          warp(warp_id).ibuffer_flush();
        } 
        else 
        {
          valid_inst = true;
          if (!m_scoreboard->checkCollision(warp_id, pI)) {
            SCHED_DPRINTF(
                "Warp (warp_id %u, dynamic_warp_id %u) passes scoreboard\n",
                (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
            ready_inst = true;

            const active_mask_t &active_mask =
                m_shader->get_active_mask(warp_id, pI);

            assert(warp(warp_id).inst_in_pipeline());

            if ((pI->op == LOAD_OP) or (pI->op == STORE_OP) or (pI->op == MEMORY_BARRIER_OP) or
                (pI->op == TENSOR_CORE_LOAD_OP) or (pI->op == TENSOR_CORE_STORE_OP)) 
            {
              if (m_mem_out->has_free(m_shader->m_config->sub_core_model, m_id) and
                  (!diff_exec_units or previous_issued_inst_exec_type != exec_unit_type_t::MEM)) 
              {
                m_shader->issue_warp(*m_mem_out, pI, active_mask, warp_id, m_id);
                issued++;
                issued_inst = true;
                warp_inst_issued = true;
                previous_issued_inst_exec_type = exec_unit_type_t::MEM;
                ///////////////////////////////// some velma stuff //////////////////////////// 
                // this is where we add the new warp_id/pc pair
                //RECORD VELMA ACCESS HERE!
                
                //Get a list of addresses, record the entries.
                std::set<new_addr_type> pI_lineaddrs = pI->get_lineaddrs();// = nc_pI.get_lineaddrs();
                std::set<velma_addr_t> vaddrs;
                for (new_addr_type lineaddr : pI_lineaddrs){
                  velma_addr_t tvaddr = static_cast<velma_addr_t>(lineaddr);
                  vaddrs.insert(tvaddr);
                  //printf("velma addr in scheduler!\n");
                  //std::cout << "velma addr in scheduler!\n";
                  //fprintf(stderr, "velma addr in scheduler!\n");
                }
                                                
                for (velma_addr_t vaddr : vaddrs){  
                  record_velma_access(warp_id, pc, vaddr);
                }
                 
              }
            } 
            else 
            {
              // This code need to be refactored
              if (pI->op != TENSOR_CORE_OP && pI->op != SFU_OP &&
                  pI->op != DP_OP && !(pI->op >= SPEC_UNIT_START_ID)) {
                bool execute_on_SP = false;
                bool execute_on_INT = false;

                bool sp_pipe_avail =
                    (m_shader->m_config->gpgpu_num_sp_units > 0) &&
                    m_sp_out->has_free(m_shader->m_config->sub_core_model,
                                       m_id);
                bool int_pipe_avail =
                    (m_shader->m_config->gpgpu_num_int_units > 0) &&
                    m_int_out->has_free(m_shader->m_config->sub_core_model,
                                        m_id);

                // if INT unit pipline exist, then execute ALU and INT
                // operations on INT unit and SP-FPU on SP unit (like in Volta)
                // if INT unit pipline does not exist, then execute all ALU, INT
                // and SP operations on SP unit (as in Fermi, Pascal GPUs)
                if (m_shader->m_config->gpgpu_num_int_units > 0 &&
                    int_pipe_avail && pI->op != SP_OP &&
                    !(diff_exec_units &&
                      previous_issued_inst_exec_type == exec_unit_type_t::INT))
                  execute_on_INT = true;
                else if (sp_pipe_avail &&
                         (m_shader->m_config->gpgpu_num_int_units == 0 ||
                          (m_shader->m_config->gpgpu_num_int_units > 0 &&
                           pI->op == SP_OP)) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::SP))
                  execute_on_SP = true;

                if (execute_on_INT || execute_on_SP) {
                  // Jin: special for CDP api
                  if (pI->m_is_cdp && !warp(warp_id).m_cdp_dummy) {
                    assert(warp(warp_id).m_cdp_latency == 0);

                    if (pI->m_is_cdp == 1)
                      warp(warp_id).m_cdp_latency =
                          m_shader->m_config->gpgpu_ctx->func_sim
                              ->cdp_latency[pI->m_is_cdp - 1];
                    else  // cudaLaunchDeviceV2 and cudaGetParameterBufferV2
                      warp(warp_id).m_cdp_latency =
                          m_shader->m_config->gpgpu_ctx->func_sim
                              ->cdp_latency[pI->m_is_cdp - 1] +
                          m_shader->m_config->gpgpu_ctx->func_sim
                                  ->cdp_latency[pI->m_is_cdp] *
                              active_mask.count();
                    warp(warp_id).m_cdp_dummy = true;
                    break;
                  } else if (pI->m_is_cdp && warp(warp_id).m_cdp_dummy) {
                    assert(warp(warp_id).m_cdp_latency == 0);
                    warp(warp_id).m_cdp_dummy = false;
                  }
                }

                if (execute_on_SP) {
                  m_shader->issue_warp(*m_sp_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::SP;
                } else if (execute_on_INT) {
                  m_shader->issue_warp(*m_int_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::INT;
                }
              } else if ((m_shader->m_config->gpgpu_num_dp_units > 0) &&
                         (pI->op == DP_OP) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::DP)) {
                bool dp_pipe_avail =
                    (m_shader->m_config->gpgpu_num_dp_units > 0) &&
                    m_dp_out->has_free(m_shader->m_config->sub_core_model,
                                       m_id);

                if (dp_pipe_avail) {
                  m_shader->issue_warp(*m_dp_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::DP;
                }
              }  // If the DP units = 0 (like in Fermi archi), then execute DP
                 // inst on SFU unit
              else if (((m_shader->m_config->gpgpu_num_dp_units == 0 &&
                         pI->op == DP_OP) ||
                        (pI->op == SFU_OP) || (pI->op == ALU_SFU_OP)) &&
                       !(diff_exec_units && previous_issued_inst_exec_type ==
                                                exec_unit_type_t::SFU)) {
                bool sfu_pipe_avail =
                    (m_shader->m_config->gpgpu_num_sfu_units > 0) &&
                    m_sfu_out->has_free(m_shader->m_config->sub_core_model,
                                        m_id);

                if (sfu_pipe_avail) {
                  m_shader->issue_warp(*m_sfu_out, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::SFU;
                }
              } else if ((pI->op == TENSOR_CORE_OP) &&
                         !(diff_exec_units && previous_issued_inst_exec_type ==
                                                  exec_unit_type_t::TENSOR)) {
                bool tensor_core_pipe_avail =
                    (m_shader->m_config->gpgpu_num_tensor_core_units > 0) &&
                    m_tensor_core_out->has_free(
                        m_shader->m_config->sub_core_model, m_id);

                if (tensor_core_pipe_avail) {
                  m_shader->issue_warp(*m_tensor_core_out, pI, active_mask,
                                       warp_id, m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type = exec_unit_type_t::TENSOR;
                }
              } else if ((pI->op >= SPEC_UNIT_START_ID) &&
                         !(diff_exec_units &&
                           previous_issued_inst_exec_type ==
                               exec_unit_type_t::SPECIALIZED)) {
                unsigned spec_id = pI->op - SPEC_UNIT_START_ID;
                assert(spec_id < m_shader->m_config->m_specialized_unit.size());
                register_set *spec_reg_set = m_spec_cores_out[spec_id];
                bool spec_pipe_avail =
                    (m_shader->m_config->m_specialized_unit[spec_id].num_units >
                     0) &&
                    spec_reg_set->has_free(m_shader->m_config->sub_core_model,
                                           m_id);

                if (spec_pipe_avail) {
                  m_shader->issue_warp(*spec_reg_set, pI, active_mask, warp_id,
                                       m_id);
                  issued++;
                  issued_inst = true;
                  warp_inst_issued = true;
                  previous_issued_inst_exec_type =
                      exec_unit_type_t::SPECIALIZED;
                }
              }
            }  // end of else
          } 
          else {
            SCHED_DPRINTF(
                "Warp (warp_id %u, dynamic_warp_id %u) fails scoreboard\n",
                (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
          }
        }
      } else if (valid) {
        // this case can happen after a return instruction in diverged warp
        SCHED_DPRINTF(
            "Warp (warp_id %u, dynamic_warp_id %u) return from diverged warp "
            "flush\n",
            (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id());
        warp(warp_id).set_next_pc(pc);
        warp(warp_id).ibuffer_flush();
      }
      if (warp_inst_issued) {
        SCHED_DPRINTF(
            "Warp (warp_id %u, dynamic_warp_id %u) issued %u instructions\n",
            (*iter)->get_warp_id(), (*iter)->get_dynamic_warp_id(), issued);
        do_on_warp_issued(warp_id, issued, iter);
      }
      checked++;
    }

    if (issued) {

      // This might be a bit inefficient, but we need to maintain
      // two ordered list for proper scheduler execution.
      // We could remove the need for this loop by associating a
      // supervised_is index with each entry in the
      // m_next_cycle_prioritized_warps vector. For now, just run through until
      // you find the right warp_id
      for (std::vector<shd_warp_t *>::const_iterator supervised_iter =
               m_supervised_warps.begin();
           supervised_iter != m_supervised_warps.end(); ++supervised_iter) {
        if (*iter == *supervised_iter) {
          m_last_supervised_issued = supervised_iter;
        }
      }
      m_num_issued_last_cycle = issued;
      if (issued == 1)
        m_stats->single_issue_nums[m_id]++;
      else if (issued > 1)
        m_stats->dual_issue_nums[m_id]++;
      else
        abort();  // issued should be > 0

      break;
    }
  }
  
  // issue stall statistics:
  if (!valid_inst)
    m_stats->shader_cycle_distro[0]++;  // idle or control hazard
  else if (!ready_inst)
    m_stats->shader_cycle_distro[1]++;  // waiting for RAW hazards (possibly due
                                        // to memory)
  else if (!issued_inst)
    m_stats->shader_cycle_distro[2]++;  // pipeline stalled
}


void velma_scheduler::order_velma_lrr(std::vector<shd_warp_t*> &reordered, 
                                      const typename std::vector<shd_warp_t*> &warps,
                                      const typename std::vector<shd_warp_t*>
                                                        ::const_iterator &just_issued,
                                      unsigned num_warps_to_add) 
{
  reordered.clear(); //clean slate.
  if (num_warps_to_add > warps.size()){
    fprintf(stderr, 
          "Number of warps to add: %d Number of warps available: %d\n", 
          num_warps_to_add, 
          warps.size());
    abort();
  }
  //

  /*We could model things in terms of iterations a bit more explicitly, but I think a different
    strategy is better. After finding our current leader and prioritizing it first, we should then 
    choose to prioritize other warps in the leader's cluster until the timer hits zero, over ALL OTHER WARPS.
    Once the timer hits zero, we revert the cluster to being non-velma?

    Either way, we fix this AFTER we know the lay of the land vis a vis configuration. 
  */
  std::vector<shd_warp_t *> non_velma_warps;
  //this probably causes the edge-case segfaults. (out of bounds) 
  auto warps_itr = (just_issued != warps.end()) ? just_issued + 1 : warps.begin();  
  for (int warps_seen = 0; warps_seen < num_warps_to_add; warps_seen++, ++warps_itr){
    warps_itr = (warps_itr != warps.end()) ? warps_itr : warps.begin(); 
    //velma check. 
    unsigned wid = (*warps_itr)->get_warp_id();
    if (is_velma_wid(wid)){
      reordered.push_back(*warps_itr);
    }
    else {
      //non_velma_warps.push_back(*warps_itr);
    }
  }
  //now push non_velma_warps to the back of reordered!
  reordered.insert(reordered.end(), non_velma_warps.begin(), non_velma_warps.end());
  non_velma_warps.clear();
}

velma_id_t velma_scheduler::record_velma_access(warp_id_t wid, velma_pc_t pc, velma_addr_t addr){
  //primary conversion to warpcluster size 
  warp_id_t wcid = wid / VELMA_WARPCLUSTER_SIZE;  
  //get our velma id. 
  /*velma_id_t vid = get_warp_pc_pair_vid(wcid,pc);
  if (vid == -1) 
  { //not tracking it? let's do that! 
    vid = add_new_velma_entry(wcid, pc);
  }

   * Need to label the $line with velma ID!. We do so 
   * by informing L1 of the new entry. Then, L1 will 
   * mark the appropriate cache line. 
   * First, though, we need access to the tag array.
   *
  //Have the cache label the line! 
  if (vid != -1){
    tag_arr->label_velma_line(vid,addr);
  }
  */ 
  return wid;//was vid before dummification for make test :)
}


velma_scheduler::velma_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
              Scoreboard *scoreboard, simt_stack **simt,
              std::vector<shd_warp_t *> *warp, register_set *sp_out,
              register_set *dp_out, register_set *sfu_out,
              register_set *int_out, register_set *tensor_core_out,
              std::vector<register_set *> &spec_cores_out,
              register_set *mem_out, int id)
    : scheduler_unit(stats, shader, scoreboard, simt, warp, sp_out, dp_out,
                     sfu_out, int_out, tensor_core_out, spec_cores_out, mem_out, id)
{ 
  //Create our pool of velma ids!  
  for (int i = 0; i < MAX_VELMA_IDS; i++){
    vid_flag_pair_t vid_pool_entry = {i, false};
    velma_id_pool.push_back(vid_pool_entry);
  }

  //initialize pointers to structures in L1 
  ldstu = m_shader->m_ldst_unit;
  mL1D = ldstu->m_L1D;
  tag_arr = mL1D->m_tag_array;

}


