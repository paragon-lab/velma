#include "velma.h"
/*#include "../abstract_hardware_model.h"
#include "addrdec.h"
#include "dram.h"
#include "gpu-cache.h"
#include "shader_trace.h"
*/


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
  wc_mask.set(warp_index); //should this be reset?
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

velma_id_t warpcluster_entry_t::get_active_velma_id(){
  velma_id_t retvid = -1;
  if (!velma_entries.empty()){
    retvid = velma_entries.begin()->velma_id;
  }
  return retvid;
}

std::set<velma_id_t> warpcluster_entry_t::all_velma_ids_from_pc(velma_pc_t pc){
  std::set<velma_id_t> vids;
  for (velma_entry_t& velma_entry : velma_entries){
    if (velma_entry.pc == pc){
      vids.insert(velma_entry.velma_id);
    }
  }
  return vids;
}

//Checks this cluster's subtable for the pc in question.
//We only care about the first instance.
velma_id_t warpcluster_entry_t::first_velma_id_from_pc(velma_pc_t pc){
  for (velma_entry_t& velma_entry : velma_entries){
    if (velma_entry.pc == pc){
      return velma_entry.velma_id;
    }
  }
  return -1;
}

//checks if a vid is present in this cluster's subtable
bool warpcluster_entry_t::contains_velma_id(velma_id_t vid){
  for (velma_entry_t& velma_entry : velma_entries){
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
  for (auto& id_flag : velma_ids_flags){
    if (id_flag.second == true){
      free_id = id_flag.first;
      break;
    }
  }
  return free_id; 
}

//marks a velma_id as not free. 
void velma_table_t::mark_velma_id_taken(velma_id_t vid){
  for (auto& id_flag : velma_ids_flags){
    if (id_flag.first == vid){
      id_flag.second = false; 
    }
  }
}
    


//finds a free velma_id, marks it as not free, and returns it. 
velma_id_t velma_table_t::get_free_velma_id(){
  velma_id_t free_id = -1;
  for (auto& id_flag : velma_ids_flags){
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
  for (auto& wc_entry : warpclusters){
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
    target = &(warpclusters.find(wcid)->second);
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
  
  velma_id_t active_vid = active_wc->get_active_velma_id();
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

  velma_id_t curr_active_vid = -1; 
  if (active_wc != nullptr) //maybe here? probably here TODO segF 
    active_wc->get_active_velma_id();
  
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
  for (auto&  id_addr : cycle_accumulated_vids_addrs){
    tag_arr->label_velma_line(id_addr.first, id_addr.second);
  }
  cycle_accumulated_vids_addrs.clear();

  //now change the active velma_id. 
  active_velma_id = active_wc->get_active_velma_id();
}



//this only sort of does what it says. really removes the ative entry. 
//returns the velma_id of the next entry in the cluster, or -1. 
velma_id_t velma_table_t::pop_dead_entry(warp_id_t wcid, velma_id_t vid){
  velma_id_t new_front_vid = -1;
  //tracking the warpcluster? 
  if (warpclusters.find(wcid) != warpclusters.end()){
    warpcluster_entry_t& wc = warpclusters[wcid];
    //tracking the vid? 
    assert(vid == wc.get_active_velma_id()); //right now, with only one vid...  
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
  velma_id_t active_vid = awc->get_active_velma_id(); 
  if (active_vid == -1) return true;
  //finally, check to see if this warp has reached. 
  return !(awc->velma_entries.begin()->has_warp_reached(wid));
}


velma_table_t::velma_table_t(int num_velma_ids){
  //populate velma id table 
  for (int i = 0; i < num_velma_ids; i++){
    velma_ids_flags.insert({static_cast<velma_id_t>(i), true});
  }
}



void velma_table_t::set_tag_array(tag_array* tag_arr_){
  tag_arr = tag_arr_;
}



