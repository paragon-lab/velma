#pragma once

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



#define VELMA_warps_per_cluster 8
#define VELMA_IDS_PER_SM 64
#define VELMA_CLUSTERS_PER_SM 2
//result from old histogramming. 
#define VELMA_temperature_START 1024


using velma_id_t = int64_t;
using velma_temperature_t = uint16_t;
using warp_id_t = unsigned; 
using velma_pc_t = unsigned; 
using velma_addr_t = uint64_t; 







//the individual velma clues in the velma_cluster clue.
struct clue_t{
    velma_pc_t pc;
    velma_id_t velma_id = -1; 
    std::vector<bool> reaching_bitmask;
    velma_temperature_t temperature;
    short warps_per_cluster;
    
     
    clue_t(velma_pc_t pc_, 
                  velma_id_t vid, 
                  velma_temperature_t temperature_start, 
                  short warps_per_cluster);

    inline void mark_warp_reached(warp_id_t wid);
    inline bool all_reached();
    inline bool has_warp_reached(warp_id_t wid);
    inline bool is_cold();

    /* Decrements the temperature. If the timer hits 0,
     * return the velma_id. Otherwise, return -1.
     */ 
    inline unsigned charge_timer();

    ~clue_t(){}
  };



//this data structure is the entire velma tracking set for 1 (one) (I) 
//velma_cluster. There will likely be more than one of these clues.
struct velma_cluster_t{
  //need both pop_front() and pop_back(), so we keep our clues in a deque.
  std::deque<clue_t> clues; 
  warp_id_t cluster_id = (unsigned)-1; 
  velma_id_t active_clue_id = -1;  

  velma_cluster_t(){}
  
  ~velma_cluster_t(){
    clues.clear();
  }

  velma_cluster_t(velma_id_t wcid){
    cluster_id = wcid;
  }


  clue_t* get_clue(velma_id_t vid);


  /* Which velma_id is the one we're currently basing
   * this velma_cluster's scheduling decisions on? 
   */ 
  velma_id_t get_active_clue_id();



  /* Pops the top velma clue, advancing the queue.
   * also returns the velma id of that element,
   * or -1 if the list is empty. */
  velma_id_t advance_queue();
  
  /* Marks the first velma clue with a matching pc in 
   * which the warp has not been marked, mark it, and 
   * return the velma id of that clue. Returns -1 if 
   * pc isn't being tracked.  
   */
  velma_id_t mark_warp_reached_pc(warp_id_t wid, velma_pc_t pc);

  
  //simply just tells us if this cluster is tracking the pc in question
  bool tracking_pc(velma_pc_t pc);

  std::vector<velma_id_t> report_expiring_vids();
  std::vector<velma_id_t> evict_cold_clues();
  velma_id_t remove_cold_clue(velma_id_t vid);

};

enum velma_status {
    VELMA_ACTIVE_NOT_REACHED,
    VELMA_NOT_REACHED,
    NON_VELMA,
    VELMA_REACHED,
    VELMA_ACTIVE_REACHED
  };

class tag_array;
class shader_core_ctx;
class velma_table_t{
  friend class velma_scheduler; 
  friend class velma_nocache_scheduler;
  friend class velru_scheduler;
  friend class tag_array;

  public:

  shader_core_ctx* shader;
  tag_array* tag_arr = nullptr; 
  int ids_per_sm;
  int warps_per_cluster;
  int clusters_per_sm;
  int temperature_start;
    
  //pointer to the active warpcluster. 
  velma_cluster_t* active_cluster = nullptr; 

  velma_table_t(){}
  ~velma_table_t(){}

  //velma_table_t(tag_array* tag_arr_, int num_velma_ids);
  velma_table_t(shader_core_ctx* m_shader, tag_array* m_tag_arr, int velma_ids_per_sm,
                            int warps_per_velma_cluster, int velma_clusters_per_sm, int velma_temperature_start);

  void reset();



  std::multimap<velma_id_t, velma_addr_t> cycle_accumulated_vids_addrs;
  std::map<warp_id_t, velma_cluster_t> velma_clusters; 
  std::map<velma_id_t, bool> velma_ids_flags;
  
  
  

  void free_velma_id(velma_id_t vid);
  velma_id_t get_free_velma_id();
  velma_id_t find_free_velma_id();
  void mark_velma_id_taken(velma_id_t vid);


  velma_id_t add_clue(velma_cluster_t* wc, velma_pc_t pc);
  velma_cluster_t* add_velma_cluster(warp_id_t wid);
  velma_id_t record_warp_access(warp_id_t wid, velma_pc_t pc);
  void record_line_access(velma_id_t vid, velma_addr_t lineaddr);                                                                  //

  void set_active_velma_cluster(warp_id_t wcid); 

  velma_cluster_t* get_velma_cluster(warp_id_t wcid);

  bool warp_active(warp_id_t wid);

  virtual void cycle();

  velma_id_t pop_cold_clue();
  velma_id_t pop_cold_clue(warp_id_t wcid, velma_id_t vid);

  bool warp_unmarked_for_active_vid(warp_id_t wid);

  virtual void set_tag_array(tag_array* tag_arr); 

  
  velma_status determine_warp_status(warp_id_t wid);

  
  void free_vids(std::vector<velma_id_t> vids); 
  bool warp_has_reached_nth_vid(int n, warp_id_t wid);
  std::vector<velma_id_t> evict_cold_clues();
  void clear_empty_clusters();

  void charge_timer(warp_id_t wid, velma_pc_t pc);


  void flush();
    
};






