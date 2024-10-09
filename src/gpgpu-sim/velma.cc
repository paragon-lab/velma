#include <string.h>

#include "velma.h"
#include "../abstract_hardware_model.h"
#include "../../libcuda/gpgpu_context.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_sim.h"
#include "../statwrapper.h"
#include "addrdec.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "gpu-sim.h"
#include "icnt_wrapper.h"
#include "mem_fetch.h"
#include "mem_latency_stat.h"
#include "shader.h"
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

//decrements the killtimer of the entry. 
inline unsigned velma_entry_t::decr_killtimer(){
  return --killtimer;  
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


/* Decrement the killtimer for the velma_entry at the top of this entry's queue. 
 * If it reaches zero, return the velma_id. Otherwise, return -1.  
 */
unsigned warpcluster_entry_t::decr_top_killtimer(){
  //no segfaults, please. 
  if (!velma_entries.empty()){ 
    //decrement the killtimer of the first element in the queue. 
    //if it's zero, we return that element's velma_id. 
    return velma_entries.begin()->decr_killtimer();
  }
  else return VELMA_KILLTIMER_START + 1;
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
 * also returns the velma id of the NEXT element 
 * or -1 if the list becomes empty.
 */
velma_id_t warpcluster_entry_t::advance_queue(){
  velma_id_t ret_vid = -1; 
  
  if(!velma_entries.empty()){
    velma_entries.pop_front(); 
  }
  //check again if the list is empty. if it isn't return the next velma id in the queue
  if(!velma_entries.empty()){
    ret_vid = velma_entries.begin()->velma_id; 
  }
  return ret_vid; 
}


velma_id_t warpcluster_entry_t::mark_warp_reached_pc(warp_id_t wid, velma_pc_t pc){
  velma_id_t marked_vid = -1;
  for (velma_entry_t& entry : velma_entries){
    //does this entry correspond to the pc we care about? 
    if (entry.pc == pc and !entry.has_warp_reached(wid)){
      //if this warp isn't marked, mark it, and return the id!
      entry.mark_warp_reached(wid);
      marked_vid = entry.velma_id;
      break;
    }
  }
  return marked_vid;
}



void warpcluster_entry_t::add_velma_entry_to_queue(velma_pc_t pc, velma_id_t vid){
    velma_entries.emplace_back(velma_entry_t(pc, vid)); 
}





/////////////////////////////////////////////////////////////////////////////////
////////////////////////    VELMA TABLE /////////////////////
//////////////////////////////////////////////////

bool velma_table_t::free_velma_id(velma_id_t vid){
  bool insertion_completed = false;
  if (vid != -1){
    assert(velma_ids_flags.find(vid) != velma_ids_flags.end());
    assert(velma_ids_flags[vid] == false); //should not duplicate frees 
    velma_ids_flags[vid] = true; 
    insertion_completed = true;
  }
  return insertion_completed;
}

//looks for a free velma_id. that's it. 
velma_id_t velma_table_t::find_free_velma_id(){
  velma_id_t free_id = -1;
  for (std::pair<velma_id_t, bool>& id_flag : velma_ids_flags){
    if (id_flag.second == true){
      free_id = id_flag.first;
      break;
    }
  }
  return free_id; 
}

//marks a velma_id as not free. 
void velma_table_t::mark_velma_id_taken(velma_id_t vid){
  for (std::pair<velma_id_t, bool>& id_flag : velma_ids_flags){
    if (id_flag.first == vid){
      id_flag.second = false; 
    }
  }
}
    


//finds a free velma_id, marks it as not free, and returns it. 
velma_id_t velma_table_t::get_free_velma_id(){
  velma_id_t free_id = -1;
  for (std::pair<velma_id_t, bool>& id_flag : velma_ids_flags){
    if (id_flag.second == true){
      free_id = id_flag.first;
      id_flag.second = false; 
      break;
    }
  }
  return free_id; 
}



void velma_table_t::record_line_access(velma_id_t vid, velma_addr_t lineaddr){
  cycle_accumulated_vids_addrs.insert({vid, lineaddr});
}


/* Records an access in the velma_table. In the case that a wcid/pc combo 
 * has already been assigned a velma id, just update the mask in the appropriate
 * velma entry and return its velma id. If the same wid/pc combo is assigned
 * multiple velma_ids, we only mark the first entry in the queue in which 
 * warp wid's bit is set high in the mask. 
 *
 * If either A. the warpcluster hasn't been tracked, B. the PC hasn't been 
 * tracked, or C. no instance of the bit unset is present, do the following:
 * IF there's space, create new velma and warpcluster entries as necessary,
 * returning the newly-assigned velma id. If there isn't space, return -1.
 */
velma_id_t velma_table_t::record_warp_access(warp_id_t wid, velma_pc_t pc){
  velma_id_t access_vid = -1; 
  warpcluster_entry_t* wc = nullptr;  
  //first: check if we're tracking the warp 
  for (std::pair<warp_id_t, warpcluster_entry_t>& wc_entry : warpclusters){
    if (wc_entry.first/VELMA_WARPCLUSTER_SIZE == wid) 
      wc = &(wc_entry.second); //nute gunray has a question 
  }

  //if we aren't tracking the warp, do we have space to?
  if (wc == nullptr and warpclusters.size() < MAX_VELMA_CLUSTERS){
    //we do! let's add a new warp 
    wc = add_warpcluster(wid);
  }
  
  if (wc != nullptr){ 
    //find and mark the first suitable velma entry. will return -1 if there isn't one. 
    access_vid = wc->mark_warp_reached_pc(wid, pc);
    //if we don't find a corresponding entry:
    if (access_vid == -1){
      access_vid = add_entry(wc, pc);
    }
  }  
  //if we touched a velma entry, return its velma_id. 
  return access_vid;
}

//if there's space, adds a new velma entry for pc to wc->velma_entries
velma_id_t velma_table_t::add_entry(warpcluster_entry_t* wc, velma_pc_t pc){
  velma_id_t free_vid = find_free_velma_id();
  //does this warpcluster have space for a new entry? did we get a velma_id? 
  if (free_vid > -1 and wc->velma_entries.size() < MAX_VELMA_IDS_PER_CLUSTER){
    //add the new entry 
    wc->add_velma_entry_to_queue(pc, free_vid);
    //since we're actually using it, mark vid as taken. 
    mark_velma_id_taken(free_vid);
  }
  return free_vid;
}

//adds a new warpcluster_entry_t to warpclusters and returns a pointer to it.
warpcluster_entry_t* velma_table_t::add_warpcluster(warp_id_t wid){
  warp_id_t wcid = wid/VELMA_WARPCLUSTER_SIZE;
  warpclusters.insert({wcid, warpcluster_entry_t(wcid)});
  warpcluster_entry_t* wc_ptr = &(warpclusters[wcid]);
  return wc_ptr;
}


void velma_table_t::set_active_warpcluster(warp_id_t wcid){
  active_wc = get_warpcluster(wcid); 
}


//returns a pointer to a given warpcluster.  
warpcluster_entry_t* velma_table_t::get_warpcluster(warp_id_t wcid){
  warpcluster_entry_t* target = nullptr;
  if (warpclusters.find(wcid) != warpclusters.end()){
    target = warpclusters.find(wcid);
  }
  return target;
}


//returns a pointer to the active warpcluster (the one being prioritized) 
warpcluster_entry_t* velma_table_t::get_active_warpcluster(){
  return active_wc;
}

/* Cycles through the active velma ids (presently only one),
 * noting all ids whose killtimers are reduced to zero. 
 */ 
velma_id_t velma_table_t::active_killtimer_cycle(){
  if (active_wc == nullptr) return -1; //-1 return to indicate no id found. 

  velma_id_t active_vid = active_wc->active_velma_id();
  if (active_wc->decr_top_killtimer() > 0){ 
    active_vid = -1; //negative 1 vid to indicate no expiration :)
  }  
  return active_vid; 
}


/* Cycles the velma_table. 
 * -killtimer decrementing
 * -set the active_velma_id to active_wc->active_velma_id
 * -handle expired velma ids 
 *    1. push expired ids to tag_arr for delabeling
 *    2. free expired ids in the velma table. 
 */ 
void velma_table_t::cycle(){
  if (warpclusters.empty()) return;
  velma_id_t curr_active_vid = active_wc->active_velma_id();
  velma_id_t next_active_vid = -1;          
  
  //killtimer decrementing
  velma_id_t expiring_vid = active_killtimer_cycle();

  //if we have a dead vid, pop it, advance the queue, and clear the expired vid in cache. 
  if (expiring_vid == curr_active_vid and expiring_vid != -1){ 
    /* The active velma id has expired. Kill it. */
    next_active_vid = pop_dead_entry(active_wc->cluster_id, expiring_vid);
    //we've advanced the queue, clear the expired id in cache!
    tag_arr->clear_expired_velma_ids({expiring_vid});
    
  }
  
  //if we have an expiration and the cluster still has entries 
  if (!active_wc->velma_entries.empty()){
    active_velma_id = next_active_vid;
  }
  else {
    //empty cluster! nuke it.
    warpclusters.erase(active_wc->cluster_id);
    if (!warpclusters.empty()){
      active_wc = &(warpclusters.begin()->second);
    }
  }

  //have the tag array label all the lines for this cycle. 
  for (std::pair<velma_id_t, velma_addr_t>&  id_addr : cycle_accumulated_vids_addrs){
    tag_arr->label_velma_line(id_addr.first, id_addr.second);
  }
  cycle_accumulated_vids_addrs.clear();

  //now change the active velma_id. 
  active_velma_id = active_wc->active_velma_id();
}



//this only sort of does what it says. really removes the ative entry. 
//returns the velma_id of the next entry in the cluster, or -1. 
velma_id_t velma_table_t::pop_dead_entry(warp_id_t wcid, velma_id_t vid){
  velma_id_t new_front_vid = -1;
  //tracking the warpcluster? 
  if (warpclusters.find(wcid) != warpclusters.end()){
    warpcluster_entry_t& wc = warpclusters[wcid];
    //tracking the vid? 
    assert(vid == wc.active_velma_id()); //right now, with only one vid...  
    new_front_vid = wc.advance_queue();
    free_velma_id(vid);
  }
  return new_front_vid; 
}



bool velma_table_t::warp_active(warp_id_t wid){
  warp_id_t wcid = wid / VELMA_WARPCLUSTER_SIZE; //relies on integer floor divide 
  warpcluster_entry_t* awc = get_active_warpcluster();
  if (awc == nullptr) return false; 

  if (wcid == awc->cluster_id) 
    return true;
  else 
    return false; 
}


bool velma_table_t::warp_unmarked_for_active_vid(warp_id_t wid){
  warp_id_t wcid = wid / VELMA_WARPCLUSTER_SIZE; 
  //get and check the active cluster's existence 
  warpcluster_entry_t* awc = get_active_warpcluster(); 
  if (awc == nullptr) return true;
  //this cluster isn't the active one. 
  if (awc->cluster_id != wcid) return true; 

  //active vid for the cluster 
  velma_id_t active_vid = awc->active_velma_id(); 
  if (active_vid == -1) return true;
  //finally, check to see if this warp has reached. 
  return !(awc->velma_entries.begin()->has_warp_reached(wid));
}


velma_table_t::velma_table_t(tag_array* tag_arr_, int num_velma_ids){
  //populate velma id table 
  for (int i = 0; i < num_velma_ids; i++){
    velma_ids_flags.insert({static_cast<velma_id_t>(i), true});
  }

  tag_arr = tag_arr_;

}






//////////////////////////////////////////////////////////////////////////////////
/////////////////////////   SCHEDULER       /////////////////////////
///////////////////////////////////////////////////
void velma_scheduler::order_warps() {
  order_velma_lrr(m_next_cycle_prioritized_warps, m_supervised_warps,
            m_last_supervised_issued, m_supervised_warps.size());
}




class scheduler_unit;
void velma_scheduler::cycle(){
  SCHED_DPRINTF("scheduler_unit::cycle()\n");
  bool valid_inst =
      false;  // there was one warp with a valid instruction to issue (didn't
              // require flush due to control hazard)
  bool ready_inst = false;   // of the valid instructions, there was one not
                             // waiting for pending register writes
  bool issued_inst = false;  // of these we issued one
                             //


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
                //////////////////////////////////////////////////////////////////////////// 
                ////////////    VELMA ACCESS RECORDING    //////////////////////////
                /////////////////////////////////////////////////////
              
                /*record 
                 * 1. the warp access
                 * 2. the individual line accesses. 
                 */ 
                velma_id_t access_vid = velma_table.record_warp_access(warp_id, pc); 

                //Get a list of addresses, record the entries.
                std::set<new_addr_type> pI_lineaddrs = pI->get_lineaddrs();// = nc_pI.get_lineaddrs();
                //std::set<velma_addr_t> vaddrs;
                for (new_addr_type lineaddr : pI_lineaddrs){
                  velma_addr_t vaddr = static_cast<velma_addr_t>(lineaddr);
                  if (access_vid != -1) velma_table.record_line_access(access_vid, vaddr);
                  //printf("velma addr in scheduler!\n");
                  //std::cout << "velma addr in scheduler!\n";
                  //fprintf(stderr, "velma addr in scheduler!\n");
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

      //cycle here. this decrements the killtimers and cleans up as necessary.  
      velma_table.cycle();

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
  
  //no segfaults, please. 
  if (num_warps_to_add > warps.size()){
    fprintf(stderr, 
          "Number of warps to add: %d Number of warps available: %d\n", 
          num_warps_to_add, 
          warps.size());
    abort();
  }
  

  std::vector<shd_warp_t *> active_velma_warps;
  std::vector<shd_warp_t *> inactive_velma_warps;
  std::vector<shd_warp_t *> non_velma_warps;
  

  /* Iterate across all the warps, Prioritizing, in this order: 
   * 1. the unmarked active velma warps 
   * 2. marked and inactive velma warps 
   * 3. non_velma_warps
   */

  //set up our warp iterator. 
  auto warps_itr = just_issued;
  if (warps_itr != warps.end() - 1){
    ++warps_itr;
  }
  else {
    warps_itr = warps.begin();
  }
  
  for (int warps_seen = 0; warps_seen < num_warps_to_add; warps_seen++, ++warps_itr){
    //loop back. 
    warps_itr = (warps_itr != warps.end()) ? warps_itr : warps.begin(); 
    
    //velma checks. 
    warp_id_t wid = (*warps_itr)->get_warp_id();
    warp_id_t wcid = wid / VELMA_WARPCLUSTER_SIZE; 
    warpcluster_entry_t* wc = get_warpcluster(wcid);
    warpcluster_entry_t* awc = get_active_warpcluster(); 
    
    //1. check for active velma warps. 
    if (warp_active(wid)){
      //if the warp has not reached yet in the current vid, push it to the active list. 
      if (warp_unmarked_for_active_vid(wid)) 
        active_velma_warps.push_back(*warps_itr);
      else //otherwise, push it to the inactive velma list. 
        inactive_velma_warps.push_back(*warps_itr);
    }
    else {
      non_velma_warps.push_back(*warps_itr);
    }
  }
  //now push!
  reordered.insert(reordered.end(), active_velma_warps.begin(), active_velma_warps.end());
  reordered.insert(reordered.end(), inactive_velma_warps.begin(), inactive_velma_warps.end());
  reordered.insert(reordered.end(), non_velma_warps.begin(), non_velma_warps.end());
  //and clean up!
  active_velma_warps.clear();
  non_velma_warps.clear();
  inactive_velma_warps.clear();
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

  //construct velma_table 
  velma_table = velma_table_t(tag_arr, MAX_VELMA_IDS);
}


