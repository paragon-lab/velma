#ifndef VELMA_H
#define VELMA_H

#include "abstract_hardware_model.h"
#include "shader.h"
#include <vector>
#include <map>
#include <queue>

class velma : public pipelined_simd_unit {
public:
    velma(const shader_core_config* config, const memory_config* mem_config, shader_core_ctx* core, memory_stats_t* stats);
    ~velma();

    void issue(warp_inst_t* inst);
    void cycle();
    bool is_busy() const;

private:
    void process_instructions();
    void handle_memory_return();
    void write_to_registers();

    bool m_busy;
    std::queue<warp_inst_t*> m_instruction_buffer;
    std::map<new_addr_type, std::vector<std::pair<unsigned, unsigned>>> m_cache_line_mapping;
    std::map<unsigned, std::map<unsigned, std::vector<unsigned>>> m_data_table;
    const shader_core_config* m_shader_config;
    const memory_config* m_mem_config;
    shader_core_ctx* m_core;
    memory_stats_t* m_stats;
};

#endif // VELMA_H

