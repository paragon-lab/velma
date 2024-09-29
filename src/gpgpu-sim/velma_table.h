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

#define VELMA_WARPCLUSTER_SIZE 4
//result from old histogramming. 
#define VELMA_KILLTIMER_START 256
#define MAX_VELMA_IDS 16


using velma_id_t = int64_t; 
using warp_id_t = unsigned; 
using velma_pc_t = unsigned; 

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
    velma_id_t decr_killtimer();

    ~velma_entry_t(){}
  };



//this data structure is the entire velma tracking set for 1 (one) (I) 
//warpcluster. There will likely be more than one of these entries.
struct warpcluster_entry_t{
  //need both pop_front() and pop_back(), so we keep our entries in a deque.
  std::deque<velma_entry_t> velma_entries; 
  warp_id_t cluster_id; 

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
   * pc isn't being tracked.
   */
  velma_id_t mark_warp_reached_pc(warp_id_t wid, velma_pc_t pc);

};


class velma_table_t{
  friend class velma_scheduler; 

  
  



};


