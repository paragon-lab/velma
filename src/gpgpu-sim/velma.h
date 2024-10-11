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
#include "gpu-cache.h"


#define VELMA_WARPCLUSTER_SIZE 4
//result from old histogramming. 
#define VELMA_KILLTIMER_START 256
#define MAX_VELMA_IDS_PER_CLUSTER 4
#define MAX_VELMA_CLUSTERS 4

using velma_id_t = int64_t; 
using warp_id_t = unsigned; 
using velma_pc_t = unsigned; 
using velma_addr_t = uint64_t; 


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
    inline unsigned charge_timer();

    ~velma_entry_t(){}
  };



//this data structure is the entire velma tracking set for 1 (one) (I) 
//warpcluster. There will likely be more than one of these entries.
struct warpcluster_entry_t{
  //need both pop_front() and pop_back(), so we keep our entries in a deque.
  std::deque<velma_entry_t> velma_entries; 
  warp_id_t cluster_id; 
  velma_id_t active_velma_id = -1;  

  warpcluster_entry_t(){}
  
  ~warpcluster_entry_t(){}

  warpcluster_entry_t(velma_id_t wcid){
    cluster_id = wcid;
  }



  velma_entry_t* get_velma_entry(velma_id_t vid);


  /* Which velma_id is the one we're currently basing
   * this warpcluster's scheduling decisions on? 
   */ 
  velma_id_t get_active_velma_id();
  
  void set_active_velma_id(velma_id_t vid);
  

  //decrements the killtimer for velma entry vid and 
  //returns the new value. 
  unsigned charge_timer(velma_id_t vid);
unsigned record_inst_issue();


  /* Pops the top velma entry, advancing the queue.
   * also returns the velma id of that element,
   * or -1 if the list is empty. */
  velma_id_t advance_queue();
  
  /* Marks the first velma entry with a matching pc in 
   * which the warp has not been marked, mark it, and 
   * return the velma id of that entry. Returns -1 if 
   * pc isn't being tracked.  
   */
  velma_id_t mark_warp_reached_pc(warp_id_t wid, velma_pc_t pc);

  void add_velma_entry_to_queue(velma_pc_t pc, velma_id_t vid);
  
  //simply just tells us if this cluster is tracking the pc in question
  bool tracking_pc(velma_pc_t pc);


};

enum velma_status {
    VELMA_ACTIVE_NOT_REACHED,
    VELMA_NOT_REACHED,
    NON_VELMA,
    VELMA_REACHED,
    VELMA_ACTIVE_REACHED
  };

class velma_table_t{
  friend class velma_scheduler; 
  

  velma_table_t(){}

  //velma_table_t(tag_array* tag_arr_, int num_velma_ids);
  velma_table_t(int num_velma_ids);


  ~velma_table_t(){}

  std::multimap<velma_id_t, velma_addr_t> cycle_accumulated_vids_addrs;
  
  std::map<warp_id_t, warpcluster_entry_t> warpclusters; 
  std::map<velma_id_t, bool> velma_ids_flags;
  
  warpcluster_entry_t* active_wc = nullptr; 
  velma_id_t active_velma_id = -1;
  
  tag_array* tag_arr = nullptr; 

  


  bool free_velma_id(velma_id_t vid);
  velma_id_t get_free_velma_id();
  velma_id_t find_free_velma_id();
  void mark_velma_id_taken(velma_id_t vid);


  velma_id_t add_velma_entry(warpcluster_entry_t* wc, velma_pc_t pc);
  warpcluster_entry_t* add_warpcluster(warp_id_t wid);
  velma_id_t record_warp_access(warp_id_t wid, velma_pc_t pc);
  void record_line_access(velma_id_t vid, velma_addr_t lineaddr);                                                                  //

  void set_active_warpcluster(warp_id_t wcid); 

  warpcluster_entry_t* get_active_warpcluster();
  warpcluster_entry_t* get_warpcluster(warp_id_t wcid);

  bool warp_active(warp_id_t wid);

  velma_id_t active_killtimer_cycle();
  void cycle();

  velma_id_t pop_dead_entry(warp_id_t wcid, velma_id_t vid);

  bool warp_unmarked_for_active_vid(warp_id_t wid);

  void set_tag_array(tag_array* tag_arr); 

  
  velma_status determine_warp_status(warp_id_t wid);

  
  bool warp_has_reached_nth_vid(int n, warp_id_t wid);

  
};

