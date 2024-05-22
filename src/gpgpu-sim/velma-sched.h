#ifndef VELMA_SCHED_H
#define VELMA_SCHED_H

#include "shader.h"
#include "velma.h"

class velma_gto_scheduler : public scheduler_unit {
public:
    velma_gto_scheduler(const shader_core_config* config, shader_core_ctx* core, memory_stats_t* stats);
    ~velma_gto_scheduler();

    void cycle() override;

private:
    velma* m_velma_unit;
    std::vector<shd_warp_t*> m_supervised_warps;
    const shader_core_config* m_config;
    shader_core_ctx* m_core;
    memory_stats_t* m_stats;
};

#endif // VELMA_SCHED_H

