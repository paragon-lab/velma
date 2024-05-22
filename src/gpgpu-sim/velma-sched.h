#ifndef VELMA_SCHED_H
#define VELMA_SCHED_H

#include "shader.h"
#include "velma.h"

class velma_gto_scheduler : public gto_scheduler {
public:
    velma_gto_scheduler(const shader_core_config* config, 
                        const memory_config* mem_config, 
                        shader_core_ctx* core, 
                        memory_stats_t* stats,
                        std::vector<shd_warp_t>* warp);
    ~velma_gto_scheduler();

    void cycle() override;
    void issue(register_set& pipe_reg_set) override;

private:
    velma* m_velma_unit; // Instance of the custom functional unit
};

#endif // VELMA_SCHED_H

