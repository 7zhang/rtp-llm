package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.springframework.context.annotation.Lazy;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Component
public class EndpointRegistry {

    private final ConcurrentHashMap<String, PrefillEndpoint> prefillEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DecodeEndpoint> decodeEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, PrefillEndpoint> pdFusionEndpoints = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, SimpleWorkerEndpoint> vitEndpoints = new ConcurrentHashMap<>();
    private final ConfigService configService;
    private final FlexlbBatchScheduler batchScheduler;
    private final BatchSchedulerReporter reporter;

    public EndpointRegistry(ConfigService configService,
                            @Lazy FlexlbBatchScheduler batchScheduler,
                            BatchSchedulerReporter reporter) {
        this.configService = configService;
        this.batchScheduler = batchScheduler;
        this.reporter = reporter;
    }

    public WorkerEndpoint get(String ipPort) {
        WorkerEndpoint ep = prefillEndpoints.get(ipPort);
        if (ep != null) {
            return ep;
        }
        ep = decodeEndpoints.get(ipPort);
        if (ep != null) {
            return ep;
        }
        ep = pdFusionEndpoints.get(ipPort);
        if (ep != null) {
            return ep;
        }
        return vitEndpoints.get(ipPort);
    }

    public WorkerEndpoint get(RoleType roleType, String ipPort) {
        if (roleType == RoleType.PREFILL) {
            return getPrefill(ipPort);
        }
        if (roleType == RoleType.DECODE) {
            return getDecode(ipPort);
        }
        if (roleType == RoleType.PDFUSION) {
            return getPdFusion(ipPort);
        }
        if (roleType == RoleType.VIT) {
            return getVit(ipPort);
        }
        return null;
    }

    public Map<String, ? extends WorkerEndpoint> getEndpoints(RoleType roleType) {
        if (roleType == RoleType.PREFILL) {
            return prefillEndpoints;
        }
        if (roleType == RoleType.DECODE) {
            return decodeEndpoints;
        }
        if (roleType == RoleType.PDFUSION) {
            return pdFusionEndpoints;
        }
        if (roleType == RoleType.VIT) {
            return vitEndpoints;
        }
        return Map.of();
    }

    public PrefillEndpoint getPrefill(String ipPort) {
        return prefillEndpoints.get(ipPort);
    }

    public DecodeEndpoint getDecode(String ipPort) {
        return decodeEndpoints.get(ipPort);
    }

    public PrefillEndpoint getPdFusion(String ipPort) {
        return pdFusionEndpoints.get(ipPort);
    }

    public SimpleWorkerEndpoint getVit(String ipPort) {
        return vitEndpoints.get(ipPort);
    }

    public WorkerEndpoint ensureEndpoint(RoleType roleType, String ipPort, WorkerStatus status) {
        if (roleType == RoleType.PREFILL) {
            return ensurePrefillEndpoint(ipPort, status);
        }
        if (roleType == RoleType.DECODE) {
            return ensureDecodeEndpoint(ipPort, status);
        }
        if (roleType == RoleType.PDFUSION) {
            return ensurePdFusionEndpoint(ipPort, status);
        }
        if (roleType == RoleType.VIT) {
            return ensureVitEndpoint(ipPort, status);
        }
        throw new IllegalArgumentException("Unsupported role: " + roleType);
    }

    public PrefillEndpoint ensurePrefillEndpoint(String ipPort, WorkerStatus status) {
        return prefillEndpoints.computeIfAbsent(ipPort,
                k -> new PrefillEndpoint(status, configService.loadBalanceConfig(), batchScheduler, reporter));
    }

    public DecodeEndpoint ensureDecodeEndpoint(String ipPort, WorkerStatus status) {
        return decodeEndpoints.computeIfAbsent(ipPort,
                k -> new DecodeEndpoint(status));
    }

    public PrefillEndpoint ensurePdFusionEndpoint(String ipPort, WorkerStatus status) {
        return pdFusionEndpoints.computeIfAbsent(ipPort,
                k -> new PrefillEndpoint(status, configService.loadBalanceConfig(), batchScheduler, reporter));
    }

    public SimpleWorkerEndpoint ensureVitEndpoint(String ipPort, WorkerStatus status) {
        return vitEndpoints.computeIfAbsent(ipPort, k -> new SimpleWorkerEndpoint(status));
    }

    /**
     * Replace prefill endpoint at given key. Closes old endpoint if present.
     * Note: This is primarily used in tests. Production code should use ensurePrefillEndpoint().
     */
    public void putPrefill(String ipPort, PrefillEndpoint endpoint) {
        PrefillEndpoint old = prefillEndpoints.put(ipPort, endpoint);
        if (old != null && old != endpoint) {
            old.close();
        }
    }

    /**
     * Replace decode endpoint at given key. Closes old endpoint if present.
     * Note: This is primarily used in tests. Production code should use ensureDecodeEndpoint().
     */
    public void putDecode(String ipPort, DecodeEndpoint endpoint) {
        DecodeEndpoint old = decodeEndpoints.put(ipPort, endpoint);
        if (old != null && old != endpoint) {
            old.close();
        }
    }

    public void putPdFusion(String ipPort, PrefillEndpoint endpoint) {
        PrefillEndpoint old = pdFusionEndpoints.put(ipPort, endpoint);
        if (old != null && old != endpoint) {
            old.close();
        }
    }

    public void putVit(String ipPort, SimpleWorkerEndpoint endpoint) {
        putSimple(vitEndpoints, ipPort, endpoint);
    }

    private void putSimple(ConcurrentHashMap<String, SimpleWorkerEndpoint> endpoints,
                           String ipPort, SimpleWorkerEndpoint endpoint) {
        SimpleWorkerEndpoint old = endpoints.put(ipPort, endpoint);
        if (old != null && old != endpoint) {
            old.close();
        }
    }

    public void close() {
        prefillEndpoints.values().forEach(WorkerEndpoint::close);
        decodeEndpoints.values().forEach(WorkerEndpoint::close);
        pdFusionEndpoints.values().forEach(WorkerEndpoint::close);
        vitEndpoints.values().forEach(WorkerEndpoint::close);
    }

    /**
     * Expose all prefill endpoints for per-worker metrics reporting.
     */
    public ConcurrentHashMap<String, PrefillEndpoint> getPrefillEndpoints() {
        return prefillEndpoints;
    }

    /**
     * Expose all decode endpoints for per-worker metrics reporting.
     */
    public ConcurrentHashMap<String, DecodeEndpoint> getDecodeEndpoints() {
        return decodeEndpoints;
    }

    public ConcurrentHashMap<String, PrefillEndpoint> getPdFusionEndpoints() {
        return pdFusionEndpoints;
    }

    public ConcurrentHashMap<String, SimpleWorkerEndpoint> getVitEndpoints() {
        return vitEndpoints;
    }

    public int getEndpointCount(RoleType roleType) {
        return getEndpoints(roleType).size();
    }

    /**
     * Trigger TTL eviction on all prefill and decode endpoints.
     *
     * @param ttlMs max age before eviction
     */
    public void evictExpiredAll(long ttlMs) {
        prefillEndpoints.values().forEach(ep -> ep.evictExpiredBatches(ttlMs));
        decodeEndpoints.values().forEach(ep -> ep.evictExpiredRequests(ttlMs));
        pdFusionEndpoints.values().forEach(ep -> ep.evictExpiredBatches(ttlMs));
    }

    /**
     * Periodic TTL eviction for all endpoints.
     * <p>Each endpoint is responsible for its own inflight lifecycle.
     * This scheduled method provides a safety-net fallback for entries
     * that were not cleaned up by {@code calibrate()} (e.g., engine crash,
     * network partition, status report delay).
     */
    @Scheduled(fixedRate = 60000L)
    public void scheduledEviction() {
        long ttlMs = configService.loadBalanceConfig().getFlexlbInflightTtlMs();
        evictExpiredAll(ttlMs);
    }
}
