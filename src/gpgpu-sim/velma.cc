#include "velma.h"
#include "gpu-cache.h"
/*#include "../abstract_hardware_model.h"
#include "addrdec.h"
#include "dram.h"
#include "shader.h"
#include "shader_trace.h"
*/



/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   clue_t   /////////////////////////
/////////////////////////////////////////////////////////////// 

clue_t::clue_t(velma_pc_t pc_, 
              velma_id_t vid, 
              velma_temperature_t temperature_start, 
              short warps_per_cluster) 
                  : pc(pc_), 
                    velma_id(vid), 
                    temperature(temperature_start),
                    warps_per_cluster(warps_per_cluster)
{  
  //initialize the velma_cluster mask to all 0s! 
  reaching_bitmask = std::vector<bool>(warps_per_cluster, false);
} 

inline void clue_t::mark_warp_reached(warp_id_t wid){
  reaching_bitmask[wid % warps_per_cluster] = true; //should this be reset?
}

inline bool clue_t::has_warp_reached(warp_id_t wid){
  return reaching_bitmask[wid % warps_per_cluster];
} 

inline bool clue_t::all_reached(){
  bool all_reached = true;
  for (int i = 0; i < warps_per_cluster; i++)
    all_reached = all_reached and reaching_bitmask[i];
  return all_reached;
}

inline bool clue_t::is_cold(){
  bool full_mask = true;
  //check the bitmask, change is_cold to false if any
  //warps have not yet reached. 
  for (int i = 0; i < warps_per_cluster; i++)
    full_mask = full_mask and reaching_bitmask[i];
  //now include the timer 
  return full_mask or (temperature <= 0);
}

//decrements the temperature of the clue. 
inline unsigned clue_t::decrease_temperature(){
  return --temperature;  
}


//////////////////////////////////////////////////////////////////////////////////////
////////////////////////////  velma_cluster_t  //////////////
/////////////////////////////////////////////////////////




clue_t* velma_cluster_t::get_clue(velma_id_t vid){
  clue_t* clue  = nullptr; 
  for (auto& vclue : clues){
    if (vclue.velma_id == vid and vid != -1){
      clue = &vclue;
      break;
    }
  }
  return clue; 
}

std::vector<clue_t*> velma_cluster_t::get_matching_entries(velma_pc_t pc){
  std::vector<clue_t*> pc_matching_entries;
  for (int i = 0; i < clues.size(); i++){
    if (clues[i].pc == pc) 
      pc_matching_entries.push_back(&(clues[i]));
  }
  return pc_matching_entries;
}

std::vector<clue_t*> velma_cluster_t::find_reached(warp_id_t wid, velma_pc_t pc){
  std::vector<clue_t*> matching_entries = get_matching_entries(pc);
  std::vector<clue_t*> reached_entries;
  for (int i = 0; i < matching_entries.size(); i++){
    if (matching_entries[i]->has_warp_reached(wid) == true){
      reached_entries.push_back(matching_entries[i]);
    }
  }
  return reached_entries;
}

clue_t* velma_cluster_t::find_last_reached(warp_id_t wid, velma_pc_t pc){
  std::vector<clue_t*> reaching = find_reached(wid, pc);
  return (reaching.empty()) ? nullptr : reaching.back();
}

/* If a warp reaches in more than one entry matching the pc, it simply cannot 
 * be a follower for the active entry. 
 */ 
/* finds the first warp in which the warp in question has not reached. 
 * this is pc-agnostic. Why? For the active entry, we want to cool 
 * the temperature every time a warp which has not reached executes
 * *any* instruction, not just the pc of that entry.
 */ 
clue_t* velma_cluster_t::find_first_unreached(warp_id_t wid){
  for (int i = 0; i < clues.size(); i++){
    if (!clues[i].has_warp_reached(wid)) return &(clues[i]);
  }
  return nullptr;
}


//has the warp reached in the active entry for this cluster? 
bool velma_cluster_t::reached_active(warp_id_t wid){
  if (!clues.empty())
    return clues.begin()->has_warp_reached(wid);
}


inline void velma_cluster_t::erase_entry(clue_t* clue){clues.erase(std::find(clues.begin(), clues.end(), *clue));}



inline velma_temperature_t velma_cluster_t::cool_clue(clue_t* clue){
  return clue->decrease_temperature();
}

inline void velma_cluster_t::mark_warp_reaching(clue_t* clue, warp_id_t wid){
  clue->mark_warp_reached(wid);
}


clue_t* velma_cluster_t::first_matching_unreached(warp_id_t wid, velma_pc_t pc){
  std::vector<clue_t*> matching_entries = get_matching_entries(pc);
  for (clue_t* entry : matching_entries){
    if (entry->has_warp_reached(wid) == false) return entry;
  }
  return nullptr;
}




//cooling. will evict if the clue hits 0!
void velma_cluster_t::mark_first_matching_unreached(warp_id_t wid, velma_pc_t pc){
  clue_t* unreached = first_matching_unreached(wid, pc);
  if (unreached != nullptr) unreached->mark_warp_reached(wid);
}


bool velma_cluster_t::warp_unreached_active(warp_id_t wid){
  if (clues.size() > 1) return clues.begin->has_warp_reached(wid);
  else return false;
}


velma_id_t velma_cluster_t::check_and_evict_cold(clue_t* clue){
  velma_id_t evicted_id = -1;
  if (clue->is_cold() and clue != nullptr){ 
    evicted_id = clue->velma_id; 
    erase_entry(clue);
  }
  return evicted_id; 
}


/* We need to do a few things here:
 *  1. 
 *
 */

clue_t* velma_cluster_t::charge_first_unreached(warp_id_t wid){
  clue_t* unreached = find_first_unreached(wid);;
  if (unreached != nullptr){
    unreached->decrease_temperature();
  }
  return unreached;
}


clue_t* velma_cluster_t::attempt_add_entry(velma_pc_t pc){
  clue_t* new_clue = nullptr; 
  velma_id_t free_vid = find_free_velma_id();
  if (free_vid != -1){
    mark_velma_id_taken(free_vid);
    clues.emplace_back(clue_t(pc, free_vid, temperature_start, warps_per_cluster));
    new_clue = &(clues.back());
  }
  return new_clue;
}


warp_access_ids_t velma_cluster_t::process_tracked_warp_access(warp_id_t wid, velma_pc_t pc){
  /*1st:  Find the first entry in which the warp is unmarked. Cool its temperature. */ 
  clue_t* charged_unreached = charge_first_unreached(wid);

  /*2nd:  Find the first entry matching PC in which the warp is unmarked. Mark
          it as reaching for that entry.*/ 
  clue_t* marked_reached = mark_first_matching_unreached(wid, pc);

  /*3rd: Check if the unreached and charged entries are the same. */
  bool charged_marked_same = marked_reached == charged_unreached and 
                        charged_unreached != nullptr; 

  /*4th: Check if either clue is cold and evict as necessary.
   *     Successful eviction returns the velma_id of the evicted element.*/

  velma_id_t charged_id = check_and_evict_cold(charged_unreached);
  velma_id_t marked_id = (charged_marked_same) ? charged_id : check_and_evict_cold(marked_reached);
  
  /*5th: If we did not mark one, and there's space, add an entry and mark it reached!*/
  clue_t* new_entry = nullptr;
  if (marked_id == -1){
    new_entry = attempt_add_entry(pc);
    if (new_entry != nullptr) new_entry->mark_warp_reached(wid);
  }

  /*6th: Package evictions. */
   eviction_ids(charged_id, marked_id);
}


 
/* Pops the top velma clue, advancing the queue.
 * also returns the velma id of the NEXT element 
 * or -1 if the list becomes empty.
 */
velma_id_t velma_cluster_t::advance_queue(){ 
  velma_id_t ret_vid = -1;
  switch(clues.size()){
    case 0: 
      break;
    default: //want size >=2 to hit here and fall through. 
      ret_vid = clues[1].velma_id;
    case 1:
      clues.pop_front();
      break;
  }
  return ret_vid; 
}


velma_id_t velma_cluster_t::mark_warp_reached_pc(warp_id_t wid, velma_pc_t pc){
  velma_id_t marked_vid = -1;
  for (clue_t& clue : clues){
    //does this clue correspond to the pc we care about? 
    if (clue.pc == pc and !clue.has_warp_reached(wid)){
      //if this warp isn't marked, mark it, and return the id!
      clue.mark_warp_reached(wid);
      marked_vid = clue.velma_id;
      break;
    }
  }
  return marked_vid;
}


std::vector<velma_id_t> velma_cluster_t::evict_cold_clues(){
  std::vector<velma_id_t> cold_vids;
  std::vector<clue_t*> cold_clues; 
  short clue_size = clues.size();
  for (auto itr = clues.begin(); itr != clues.end();){
    if (itr->is_cold()){
      cold_vids.push_back(itr->velma_id); 
      itr = clues.erase(itr);
    }
    else itr++;
  }
  return cold_vids;
}



/* Cycles through the active velma ids (presently only one),
 * noting all ids whose temperatures are reduced to zero. 
 */ 
std::vector<velma_id_t> velma_cluster_t::report_expiring_vids(){
  std::vector<velma_id_t> expiring_vids;
  for (auto& clue : clues){
    if (clue.temperature <= 0 )
      expiring_vids.push_back(clue.velma_id);
  }
  return expiring_vids;
}


/*
velma_id_t velma_cluster_t::evict_cold_clue(velma_id_t vid){
  clue_t* cold_clue = get_clue(vid);
  if (cold_clue != nullptr){
    clues.erase(cold_clue);
  }
  if (!clues.empty())
    return clues[0].velma_id;
  else return -1;
}
*/

/////////////////////////////////////////////////////////////////////////////////
////////////////////////    VELMA TABLE /////////////////////
//////////////////////////////////////////////////

void velma_table_t::free_velma_id(velma_id_t vid){
    velma_ids_flags[vid] = true; 
  }

//looks for a free velma_id. that's it. 
velma_id_t velma_table_t::find_free_velma_id(){
  for (auto& id_flag : velma_ids_flags){
    if (id_flag.second == true){
      return id_flag.first;
    }
  }
  return -1; //no vid found 
}

//marks a velma_id as not free. 
void velma_table_t::mark_velma_id_taken(velma_id_t vid){
  velma_ids_flags[vid] = false;
}
    


//finds a free velma_id, marks it as not free, and returns it. 
velma_id_t velma_table_t::get_free_velma_id(){
  for (auto& id_flag : velma_ids_flags){
    if (id_flag.second == true){
      id_flag.second = false; 
      return id_flag.first;
    }
  }
}



void velma_table_t::record_line_access(velma_id_t vid, velma_addr_t lineaddr){
  cycle_accumulated_vids_addrs.insert({vid, lineaddr});
}


/* Records an access in the velma_table. In the case that a wcid/pc combo 
 * has already been assigned a velma id, just update the mask in the appropriate
 * velma clue and return its velma id. If the same wid/pc combo is assigned
 * multiple velma_ids, we only mark the first clue in the queue in which 
 * warp wid's bit is set high in the mask. 
 *
 * If either A. the velma_cluster hasn't been tracked, B. the PC hasn't been 
 * tracked, or C. no instance of the bit unset is present, do the following:
 * IF there's space, create new velma and velma_cluster entries as necessary,
 * returning the newly-assigned velma id. If there isn't space, return -1.
 */
velma_id_t velma_table_t::record_warp_access(warp_id_t wid, velma_pc_t pc){
  velma_id_t access_vid = -1; 
  velma_cluster_t* wc = get_velma_cluster(wid / warps_per_cluster);

  //if we aren't tracking the warp, do we have space to?
  if (wc == nullptr and velma_clusters.size() < clusters_per_sm){
    //we do! let's add a new warp 
    wc = add_velma_cluster(wid);
  }
  
  if (wc != nullptr){ 
    //find and mark the first suitable velma clue. will return -1 if there isn't one. 
    access_vid = wc->mark_warp_reached_pc(wid, pc);
    //if we don't find a corresponding clue:
    if (access_vid == -1){
      access_vid = add_clue(wc, pc);
    }
    if (active_cluster == nullptr) active_cluster = wc;
  }  
  //if we touched a velma clue, return its velma_id. 
  return access_vid;
}

//if there's space, adds a new velma clue for pc to wc->clues
velma_id_t velma_table_t::add_clue(velma_cluster_t* wc, velma_pc_t pc){
  velma_id_t free_vid = find_free_velma_id();
  //does this velma_cluster have space for a new clue? did we get a velma_id? 
  if (free_vid > -1 and wc->clues.size() < ids_per_sm / clusters_per_sm){  
    //add the new clue 
    wc->clues.emplace_back(clue_t(pc, free_vid, temperature_start, warps_per_cluster));
    //since we're actually using it, mark vid as taken. 
    mark_velma_id_taken(free_vid);
  }
  return free_vid;
}

//adds a new velma_cluster_t to velma_clusters and returns a pointer to it.
velma_cluster_t* velma_table_t::add_velma_cluster(warp_id_t wid){
  warp_id_t wcid = wid/warps_per_cluster;
  velma_clusters.insert({wcid, velma_cluster_t(wcid)});
  velma_cluster_t* wc_ptr = &(velma_clusters[wcid]);
  return wc_ptr;
}


void velma_table_t::set_active_velma_cluster(warp_id_t wcid){
  active_cluster = get_velma_cluster(wcid); 
}


//returns a pointer to a given velma_cluster.  
inline velma_cluster_t* velma_table_t::get_velma_cluster(warp_id_t wcid){
  if (velma_clusters.find(wcid) != velma_clusters.end())
    return &(velma_clusters[wcid]);
  else return nullptr;
}






std::vector<velma_id_t> velma_table_t::evict_cold_clues(){
  std::vector<velma_id_t> cold_ids; 
  for (auto& wc : velma_clusters){
    std::vector<velma_id_t> cl_cold_ids = wc.second.evict_cold_clues();
    for (velma_id_t vid : cl_cold_ids){
      cold_ids.push_back(vid);
    }
  }
  return cold_ids;
}
 

void velma_table_t::free_vids(std::vector<velma_id_t> vids){
  for (auto& vid : vids)
    free_velma_id(vid);
}


/* Cycles the velma_table. 
 * -set the active_velma_id to active_cluster->active_velma_id
 * -handle expired velma ids 
 *    1. push expired ids to tag_arr for delabeling
 *    2. free expired ids in the velma table. 
 */ 
void velma_table_t::cycle(){
  //is the table empty? clear all the things. 

  //handle velma_id expirations 
  std::vector<velma_id_t> expiring_vids = evict_cold_clues();
  free_vids(expiring_vids);
  tag_arr->clear_expired_velma_ids(expiring_vids);    

  //clear empty clusters 
  clear_empty_clusters();
  

  if (!velma_clusters.empty()){
    //have the tag array label all the lines for this cycle. 
    for (auto&  id_addr : cycle_accumulated_vids_addrs){
      tag_arr->label_velma_line(id_addr.first, id_addr.second);
    }
  }
    
  cycle_accumulated_vids_addrs.clear();
}



//this only sort of does what it says. really removes the ative clue. 
//returns the velma_id of the next clue in the cluster, or -1. 
velma_id_t velma_table_t::pop_cold_clue(warp_id_t wcid, velma_id_t vid){
  velma_id_t new_front_vid = -1;
  //tracking the velma_cluster? 
  if (velma_clusters.find(wcid) != velma_clusters.end()){
    velma_cluster_t& wc = velma_clusters[wcid];
    //tracking the vid? 
    new_front_vid = wc.advance_queue();
    free_velma_id(vid);
  }
  return new_front_vid; 
}



bool velma_table_t::warp_active(warp_id_t wid){
  warp_id_t wcid = wid / warps_per_cluster; //relies on integer floor divide 
  if (active_cluster == nullptr) return false; 
  else return wcid == active_cluster->cluster_id;
}


bool velma_table_t::warp_has_reached_nth_vid(int n, warp_id_t wid){
  warp_id_t wcid = wid / warps_per_cluster;
  //are we even tracking this warp?
  if (velma_clusters.find(wcid) == velma_clusters.end()) return false; 
  
  velma_cluster_t* wc = &(velma_clusters.begin()->second); 
  //does this cluster HAVE n velma entries? 
  if (wc->clues.size() <= n) return false;

  //is this warp's bitmask marked in the nth clue? 
  if (wc->clues[n].has_warp_reached(wid))
    return true;
  else 
    return false; 
}

bool velma_table_t::warp_unmarked_for_active_vid(warp_id_t wid){
  warp_id_t wcid = wid / warps_per_cluster; 
  //get and check the active cluster's existence 
  if (active_cluster != nullptr and 
      active_cluster->cluster_id == wcid and 
      !active_cluster->clues.empty())
  {
    return !(active_cluster->clues.begin()->has_warp_reached(wid));
  }
  else return true; 

}



velma_table_t::velma_table_t(shader_core_ctx* m_shader, tag_array* m_tag_arr, int velma_ids_per_sm,
                            int warps_per_velma_cluster, int velma_clusters_per_sm, int velma_temperature_start) 
                                : shader(m_shader), 
                                  tag_arr(m_tag_arr), 
                                  ids_per_sm(velma_ids_per_sm),
                                  warps_per_cluster(warps_per_velma_cluster), 
                                  clusters_per_sm(velma_clusters_per_sm),
                                  temperature_start(velma_temperature_start)
{
  //populate velma id table 
  for (int i = 0; i < velma_ids_per_sm; i++){
    velma_ids_flags.insert({static_cast<velma_id_t>(i), true});
  }
}



void velma_table_t::set_tag_array(tag_array* tag_arr_){
  assert(tag_arr_);
  tag_arr = tag_arr_;
  tag_arr->velma_table = this;
}



velma_status velma_table_t::determine_warp_status(warp_id_t wid){
  //is this warp in the active velma cluster? 
  if (warp_active(wid)){
    //has this warp seen the load in the bitmask? 
    if (warp_unmarked_for_active_vid(wid)) 
      return VELMA_ACTIVE_NOT_REACHED;
    else      
      return VELMA_ACTIVE_REACHED;
  } //is this a velma warp? 
  else if (velma_clusters.find(wid / warps_per_cluster) != velma_clusters.end())
  { 
    //has this reached in its first velma clue? 
    if (!warp_has_reached_nth_vid(0, wid))
      return VELMA_NOT_REACHED; 
    else 
      return VELMA_REACHED;
  }
  else {
    return NON_VELMA;
  }
}


void velma_table_t::flush(){
  //YEET ALL THE THINGS!
  //deleta all of our tracking 
  velma_clusters.clear();
  cycle_accumulated_vids_addrs.clear();
  //reset our variables 
  active_cluster = nullptr; 
  //free all of our velma ids 
  for (auto& vid_flag : velma_ids_flags){
    vid_flag.second = true;
  }
}

warp_id_t velma_table_t::warp_id_to_cluster_id(warp_id_t wid){
  return wid / warps_per_cluster;
}


void velma_table_t::clear_empty_clusters(){
  std::vector<warp_id_t> empty_wc_ids;
  for (auto itr = velma_clusters.begin(); itr != velma_clusters.end();){    
    if (itr->second.clues.empty()){ 
      if (active_cluster == &(itr->second))
        active_cluster = nullptr;

      itr = velma_clusters.erase(itr);  
    }
    else itr++;
  }
}


//nocache_velma_table_t::nocache_velma_table_t(int num_velma_ids) : velma_table_t(num_velma_ids) {}


    

/////////////////////////////////////////////////////////////////////////////////
/////////     THIS IS THE ONE WE USE //////////////////////// 
////////////////////////////////////////////////

//get a vid from wid and pc 
void velma_table_t::cool_clue_temperature(warp_id_t wid, velma_pc_t pc){
  velma_status vstatus = determine_warp_status(wid);
  velma_cluster_t* containing_cluster = get_velma_cluster(wid / warps_per_cluster);
  switch (vstatus) {
    case VELMA_ACTIVE_NOT_REACHED: 
      active_cluster->clues.begin()->decrease_temperature();
      break;
    case VELMA_NOT_REACHED:
      get_velma_cluster(wid / warps_per_cluster)->clues.begin()->decrease_temperature();
      break;
    default: 
      break;
  }
}


void velma_table_t::reset(){
  velma_clusters.clear();
  
  for (auto& id_fl : velma_ids_flags){
    id_fl.second = 1;
  }
  active_cluster = nullptr;
}




