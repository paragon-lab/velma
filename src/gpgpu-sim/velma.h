#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <iostream>
#include "shader.h"


#define VELMA_WARPCLUSTER_SIZE 4
//result from old histogramming. 
#define VELMA_KILLTIMER_START 256
#define MAX_VELMA_IDS_PER_CLUSTER 4
#define MAX_VELMA_CLUSTERS 4

using velma_id_t = int64_t; 
using warp_id_t = unsigned; 
using velma_pc_t = unsigned; 


//the individual velma entries in the warpcluster entry.
struct velma_entry_t{
    velma_pc_t pc;
    velma_id_t velma_id = -1; 
    std::bitset<VELMA_WARPCLUSTER_SIZE> wc_mask;
    unsigned killtimer;
     
    velma_entry_t(velma_pc_t pc_, velma_id_t vid);

    inline void mark_warp_reached(warp_id_t wid);

    inline bool has_warp_reached(warp_id_t wid);

    /* Decrements the killtimer. If the timer hits 0,
     * return the velma_id. Otherwise, return -1.
     */ 
    inline unsigned decr_killtimer();

    ~velma_entry_t(){}
  };



//this data structure is the entire velma tracking set for 1 (one) (I) 
//warpcluster. There will likely be more than one of these entries.
struct warpcluster_entry_t{
  //need both pop_front() and pop_back(), so we keep our entries in a deque.
  std::deque<velma_entry_t> velma_entries; 
  warp_id_t cluster_id; 
  std::map<velma_id_t, bool> velma_ids_flags;




  warpcluster_entry_t(){}

  ~warpcluster_entry_t(){
    velma_entries.clear();
  }

  warpcluster_entry_t(velma_id_t wcid){
    cluster_id = wcid;
  }


  /* Which velma_id is the one we're currently basing
   * this warpcluster's scheduling decisions on? 
   */ 
  velma_id_t active_velma_id();

  /* Checks this cluster's subtable for the pc in question.
   * We only care about the first instance.
   * Returns -1 if that pc isn't being tracked anywhere
   */
  velma_id_t first_velma_id_from_pc(velma_pc_t pc);

  //Returns a set of every velma_id associated with the pc.
  std::set<velma_id_t> all_velma_ids_from_pc(velma_pc_t pc);

  //checks if a vid is present in this cluster's subtable
  bool contains_velma_id(velma_id_t vid);

  //decrements the top killtimer and returns the associated vid. 
  velma_id_t decr_top_killtimer();

  //reports the set of velma ids this cluster is tracking. 
  std::set<velma_id_t> report_velma_ids();

  /* Pops the top velma entry, advancing the queue.
   * also returns the velma id of that element,
   * or -1 if the list is empty. */
  velma_id_t pop_front_velma_entry();
  
  /* Marks the first velma entry with a matching pc in 
   * which the warp has not been marked, mark it, and 
   * return the velma id of that entry. Returns -1 if 
   * pc isn't being tracked. TODO: extend to try to add 
   * a new entry if there isn't an appropriate one.
   */
  velma_id_t mark_warp_reached_pc(warp_id_t wid, velma_pc_t pc);

  void add_velma_entry_to_queue(velma_pc_t pc, velma_id_t vid);
  
  //simply just tells us if this cluster is tracking the pc in question
  bool tracking_pc(velma_pc_t pc);


};


class velma_table_t{
  friend class velma_scheduler; 
  
  
  std::map<warp_id_t, warpcluster_entry_t> warpclusters; 
  std::map<velma_id_t, bool> velma_ids_flags;
  warp_id_t active_wc_id = -1; 

  bool free_velma_id(velma_id_t vid);
  velma_id_t get_free_velma_id();
  velma_id_t find_free_velma_id();
  void mark_velma_id_taken(velma_id_t vid);


  velma_id_t add_entry(warpcluster_entry_t* wc, velma_pc_t pc);
  warpcluster_entry_t* add_warpcluster(warp_id_t wid);
  velma_id_t record_access(warp_id_t wid, velma_pc_t pc);

  //TODO:something to decrement the active cluster's killtimer.
  void set_active_warpcluster(warp_id_t wcid);
  warpcluster_entry_t* get_active_warpcluster();
  warpcluster_entry_t* get_warpcluster(warp_id_t wcid);
};


class velma_scheduler : public scheduler_unit {
 public:
  /* We need to map velma ids to warp clusters and vice versa. 
   * One velma id will correspond to one and only one warpcluster id/pc combo. 
   * A warpcluster id can correspond to 0 or more velma_ids. 
   */

  //
  using velma_warp_pc_pair_t = std::pair<warp_id_t, velma_pc_t>;
  std::map<velma_id_t, velma_warp_pc_pair_t> velma_ids_pairs;
  std::map<velma_warp_pc_pair_t, velma_id_t> velma_pairs_ids;
  std::map<velma_id_t, unsigned> velma_ids_killtimers; 

  //SCHEDULING BITMASKS  
  using vid_wc_bitmask_pair_t = std::pair<velma_id_t, std::bitset<VELMA_WARPCLUSTER_SIZE>>;
  using vid_bitmask_queue_t = std::deque<vid_wc_bitmask_pair_t>;

  using wc_bitmask_queue_t = std::deque<std::bitset<VELMA_WARPCLUSTER_SIZE>>;
  using wc_pc_queue_t = std::deque<velma_pc_t>;
  

  /* We need to limit the number of velma ids we allow. 
   * To do so, we create a pool of ids, each of which is 
   * associated with a flag indicating whether or not 
   * the velma id is currently in use. False indicates that
   * the velma_id is unoccupied!
   */
  using vid_flag_pair_t = std::pair<velma_id_t, bool>;
  std::vector<vid_flag_pair_t>  velma_id_pool;

  /* We need to keep a pointer to the velma_tag_array so that 
   * the velma scheduler has direct control over the tag_array's
   * data structures. 
   */
  class ldst_unit* ldstu; 
  l1_cache* mL1D; 
  tag_array* tag_arr;
    

 
  int velma_id_ctr; 
  
  velma_scheduler(shader_core_stats *stats, shader_core_ctx *shader,
                Scoreboard *scoreboard, simt_stack **simt,
                std::vector<shd_warp_t *> *warp, register_set *sp_out,
                register_set *dp_out, register_set *sfu_out,
                register_set *int_out, register_set *tensor_core_out,
                std::vector<register_set *> &spec_cores_out,
                register_set *mem_out, int id);

  virtual ~velma_scheduler() {}
  virtual void order_warps();
  virtual void done_adding_supervised_warps() {
    m_last_supervised_issued = m_supervised_warps.end();
  }

//replaced templated version wth shd_warp_t*. same functionality.
 void order_velma_lrr(std::vector<shd_warp_t *> &reordered, 
                      const typename std::vector<shd_warp_t *> &warps,
                      const typename std::vector<shd_warp_t *> 
                                        ::const_iterator &just_issued,
                      unsigned num_warps_to_add);

  
//TODO
 bool is_velma_wid(warp_id_t wid){
   warp_id_t wcid = wid / VELMA_WARPCLUSTER_SIZE;
   return false;//velma_cluster_bitmasks.find(wcid) != velma_cluster_bitmasks.end();
 }
//TODO 
 velma_id_t get_warp_pc_pair_vid(warp_id_t wcid, velma_pc_t vpc){
   //make our pair for map indexing 
  velma_warp_pc_pair_t wcid_vpc = {wcid, vpc};
  //if we aren't tracking this velma id, just return 0.
  return -1;//(velma_pairs_ids.find(wcid_vpc) != velma_pairs_ids.end()) ? velma_pairs_ids[wcid_vpc] : -1; 
  }
 //TODO???
 velma_id_t get_free_velma_id(){
    velma_id_t vid = -1; 
    for (auto vfp : velma_id_pool){
      if (vfp.second == false){
        vid = vfp.first;
        vfp.second = true;
        break; 
      }
    }
    return vid; 
 }

 //TODO???
 bool is_free_velma_id(velma_id_t vid){
    return !velma_id_pool[vid].second; 
 }

 //TODO???
 //lots of room for optimization in velma_id_pool, but whatever.
 //keep it simple right now.
 void free_velma_id(velma_id_t vid){
    if (vid < velma_id_pool.size() and vid >= 0){
      //can just index using the vid. 
      velma_id_pool[vid].second = false;
    }
 }

 //TODO
 velma_id_t add_new_velma_entry(warp_id_t wcid, velma_pc_t vpc){
    //fprintf(stdout, "\n\nVELMA ENTRY ADDED!\n\n");
    //first thing's first: get our new vid! 
    velma_id_t vid = get_free_velma_id();
    //if there are none, we return -1. 
    if (vid == -1) return vid; 
    //make our warpid/pc pair 
    velma_warp_pc_pair_t wcid_pc = {wcid, vpc};
    //wcid_pc insertion.
    velma_pairs_ids.insert({wcid_pc, vid});
    //populate the inverse map 
    velma_ids_pairs.insert({vid, wcid_pc});
    //populate the killtimers 
    velma_ids_killtimers.insert({vid, VELMA_KILLTIMER_START});

    /* Processing the changes to our warpcluster_id->velma_id_ct map
     * isn't the simplest thing. A new vid does not necessarily mean 
     * a new warp, but the contrapositive applies*/ 
    //auto widct_find = velma_wcids_vid_cts.find(wcid);
    //if this wcid isn't present, create a row for it. 
    //if (widct_find == velma_wcids_vid_cts.end()){
    //  velma_wcids_vid_cts.insert({wcid, 0});
    //}
    //either way, now we increment the count. 
    //velma_wcids_vid_cts[wcid]++;
    return -1;//vid; 
  }

  //TODO  
  bool clear_velma_entry(velma_id_t vid){
    velma_warp_pc_pair_t wcid_pc = {0,0};
    bool vid_was_cleared = true;
    //erase pertinent entries in velma_ids_pairs
    wcid_pc = velma_ids_pairs[vid]; 
    velma_ids_pairs.erase(vid);

    //erase pertinent entries in velma_pairs_ids
    velma_pairs_ids.erase(wcid_pc);

    //erase pertinent entries in velma_ids_killtimers 
    velma_ids_killtimers.erase(vid);
    
    unsigned wcid = wcid_pc.first; 
   /*
    if (velma_wcids_vid_cts.find(wcid) != velma_wcids_vid_cts.end()){
      //does this warpid still correspond to any velma ids? 
      if (velma_wcids_vid_cts[wcid] <= 1){  
        //no? stop counting it. 
        velma_wcids_vid_cts.erase(wcid);
      }
      else {
        //yes? decrement it. 
        velma_wcids_vid_cts[wcid]--;
      }
    }*/

    //free the velma_id in the velma_id_pool
    free_velma_id(vid);

    //finally, free the corresponding lines in the tag array. 
    tag_arr->release_velma_id_lines(vid);

    return vid_was_cleared;
  }
  
  
  void reset_vid_killtimer(velma_id_t vid){
    auto findres = velma_ids_killtimers.find(vid);
    if (findres != velma_ids_killtimers.end()){
      velma_ids_killtimers[vid] = VELMA_KILLTIMER_START; 
    }
    else { //i don't think this branch will ever be taken.
      velma_ids_killtimers.insert({vid,VELMA_KILLTIMER_START});
    }
  }


  //TODO  
  /* Big stuff here. This is how we will process ld/st instructions isssued.
   * First: Calculate the warpcluster id from the warp id. 
   * Second: check if velma already knows this warpcluster and pc. 
   * If so, great! Use the velma ID to reset the appropriate killtimer. 
   * If not, we create a new velma entry. This function is also where we 
   * arbitrate the generation and distribution of velma_ids. 
   */
  velma_id_t record_velma_access(warp_id_t wid, velma_pc_t pc, velma_addr_t addr);


  //TODO
  std::set<velma_id_t> decr_pc_killtimers(){
    std::set<velma_id_t> expiring;
    for (auto& vid_kt : velma_ids_killtimers){
      //is the killtimer 0 
      if (vid_kt.second <= 0){ //yes? record the vid as expired.
        expiring.insert(vid_kt.first);
      }
      else{ //no? decrement the timer. 
        vid_kt.second--; 
      }
    }
    return expiring; 
  }

  void velma_cycle();  
  void cycle();
};



