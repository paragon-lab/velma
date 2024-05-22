#ifndef VELMA_H
#define VELMA_H

#include "shader.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>

// Hash function for pair<int, int> to be used in unordered_set
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2> &pair) const {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        return hash1 ^ hash2;
    }
};

class velma : public pipelined_simd_unit {
public:
    velma(const shader_core_config* config, 
          const memory_config* mem_config, 
          shader_core_ctx* core, 
          memory_stats_t* stats);
    ~velma();

    // Override the issue function to buffer instructions
    virtual void issue(warp_inst_t* inst);

    // Method to handle data return
    void data_return(warp_inst_t* inst, const std::vector<uint64_t>& data);

private:
    std::vector<warp_inst_t*> instruction_buffer; // Buffer for instructions
    std::unordered_map<int, std::unordered_map<int, uint64_t>> data_table; // Warp ID -> Thread ID -> Data
    std::unordered_map<warp_inst_t*, bool> request_status; // Track memory request completion

    // Cache line tracking structures
    std::unordered_map<uint64_t, std::unordered_set<std::pair<int, int>, pair_hash>> cache_line_accesses; // Cache line -> Set of (warp ID, thread ID)
    std::unordered_map<uint64_t, std::vector<uint64_t>> cache_line_data; // Cache line -> Data vector

    // Function to process the buffered instructions
    void process_batch();
    
    // Function to write data back to the register file
    void writeback_to_register(warp_inst_t* inst, const std::vector<uint64_t>& data);
    
    // Function to check if all data has been returned
    bool all_requests_complete();
    
    // Function to calculate unique cache lines and thread accesses
    void calculate_cache_lines();
};

#endif // VELMA_H

