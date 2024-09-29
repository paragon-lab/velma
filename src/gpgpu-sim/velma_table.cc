#include "velma_table.h"



/////////////////////////////////   velma_entry_t        ////////////////////////////////

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



//////////////////////////////////  warpcluster_entry_t  ////////////////////////////////

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
