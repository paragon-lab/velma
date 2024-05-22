#include "velma.h"

velma::velma(const shader_core_config* config, 
             const memory_config* mem_config, 
             shader_core_ctx* core, 
             memory_stats_t* stats)
    : pipelined_simd_unit(config, core, 8, stats, 1) {
    // Initialization code if needed
}

velma::~velma() {
    // Cleanup code if needed
}

void velma::issue(warp_inst_t* inst) {
    // Add the instruction to the buffer
    instruction_buffer.push_back(inst);
    // Mark the request as not complete
    request_status[inst] = false;

    // Check if the buffer has reached the threshold of 8 instructions
    if (instruction_buffer.size() == 8) {
        // Process the batch of instructions
        process_batch();
    }
}

void velma::process_batch() {
    // Calculate unique cache lines and their accesses
    calculate_cache_lines();

    // Issue one memory request per unique cache line
    for (const auto& entry : cache_line_accesses) {
        uint64_t cache_line = entry.first;
        // Create a memory request for this cache line
        warp_inst_t* dummy_inst = new warp_inst_t(); // Create a dummy instruction for the request
        dummy_inst->set_cache_line(cache_line); // Assuming set_cache_line sets the address to the cache line address
        dummy_inst->accessq_push_back(dummy_inst);
    }
}

void velma::calculate_cache_lines() {
    cache_line_accesses.clear();
    for (auto inst : instruction_buffer) {
        for (unsigned i = 0; i < inst->warp_size(); ++i) {
            if (inst->active(i)) {
                uint64_t address = inst->get_address(i);
                uint64_t cache_line = address & ~0x7F; // Calculate the cache line address
                int warp_id = inst->warp_id();
                int thread_id = inst->get_thread_id(i);
                cache_line_accesses[cache_line].emplace(warp_id, thread_id);
            }
        }
    }
}

void velma::data_return(warp_inst_t* inst, const std::vector<uint64_t>& data) {
    uint64_t cache_line = inst->get_cache_line(); // Assuming get_cache_line retrieves the cache line address
    cache_line_data[cache_line] = data;

    // Mark the request as complete
    for (auto& entry : request_status) {
        if (entry.first->get_cache_line() == cache_line) {
            entry.second = true;
        }
    }

    // Check if all data has been returned for the buffered instructions
    if (all_requests_complete()) {
        // Distribute data to threads and write back to registers
        for (auto inst : instruction_buffer) {
            std::vector<uint64_t> thread_data(inst->warp_size(), 0);
            for (unsigned i = 0; i < inst->warp_size(); ++i) {
                if (inst->active(i)) {
                    uint64_t address = inst->get_address(i);
                    uint64_t cache_line = address & ~0x7F;
                    int warp_id = inst->warp_id();
                    int thread_id = inst->get_thread_id(i);
                    thread_data[i] = cache_line_data[cache_line][thread_id];
                }
            }
            writeback_to_register(inst, thread_data);
        }
        instruction_buffer.clear();
        request_status.clear();
        cache_line_accesses.clear();
        cache_line_data.clear();
    }
}

void velma::writeback_to_register(warp_inst_t* inst, const std::vector<uint64_t>& data) {
    // Iterate over each active thread in the warp
    for (unsigned i = 0; i < inst->warp_size(); ++i) {
        if (inst->active(i)) {
            // Write data back to the register file
            int reg_id = inst->out[i]; // Assuming inst->out contains the destination registers
            m_core->set_reg(inst->warp_id(), reg_id, data[i]);
        }
    }
}

bool velma::all_requests_complete() {
    // Check if all requests in the batch are complete
    for (const auto& entry : request_status) {
        if (!entry.second) {
            return false;
        }
    }
    return true;
}

