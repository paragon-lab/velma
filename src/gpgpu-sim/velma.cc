#include "velma.h"
#include "shader.h"

velma::velma(const shader_core_config* config, const memory_config* mem_config, shader_core_ctx* core, memory_stats_t* stats)
    : pipelined_simd_unit(config), m_shader_config(config), m_mem_config(mem_config), m_core(core), m_stats(stats), m_busy(false) {
}

velma::~velma() {
}

void velma::issue(warp_inst_t* inst) {
    m_instruction_buffer.push(inst);
    if (m_instruction_buffer.size() == 8) {
        m_busy = true;
        process_instructions();
    }
}

void velma::cycle() {
    if (m_busy) {
        handle_memory_return();
        if (m_instruction_buffer.empty()) {
            m_busy = false;
        }
    }
}

bool velma::is_busy() const {
    return m_busy;
}

void velma::process_instructions() {
    // Clear previous cache line mapping
    m_cache_line_mapping.clear();

    // Process each instruction in the buffer
    while (!m_instruction_buffer.empty()) {
        warp_inst_t* inst = m_instruction_buffer.front();
        m_instruction_buffer.pop();

        // Iterate over threads in the warp
        for (unsigned t = 0; t < m_shader_config->warp_size; ++t) {
            if (inst->active(t)) {
                new_addr_type addr = inst->get_addr(t);
                new_addr_type cache_line = addr & ~(m_mem_config->line_size - 1);
                m_cache_line_mapping[cache_line].emplace_back(inst->warp_id(), t);
            }
        }
    }

    // Issue memory requests for each unique cache line
    for (const auto& entry : m_cache_line_mapping) {
        new_addr_type cache_line = entry.first;
        m_core->memory_system->issue_request(cache_line);
    }
}

void velma::handle_memory_return() {
    // Check if memory returns are completed
    if (m_core->memory_system->all_memory_responses_ready()) {
        for (const auto& entry : m_cache_line_mapping) {
            new_addr_type cache_line = entry.first;
            const std::vector<std::pair<unsigned, unsigned>>& threads = entry.second;
            //route data to appropriate threads 
            for (const auto& [warp_id, thread_id] : threads) {
                unsigned data = m_core->memory_system->get_data(cache_line, thread_id);
                m_data_table[warp_id][thread_id].push_back(data);
            }
        }

        write_to_registers();
    }
}

void velma::write_to_registers() {
    for (const auto& warp_entry : m_data_table) {
        unsigned warp_id = warp_entry.first;
        for (const auto& thread_entry : warp_entry.second) {
            unsigned thread_id = thread_entry.first;
            const std::vector<unsigned>& data_list = thread_entry.second;

            for (unsigned data : data_list) {
                m_core->get_shader_warp(warp_id)->set_register(thread_id, data);
            }
        }
    }

    // Clear data table after writing
    m_data_table.clear();
}

