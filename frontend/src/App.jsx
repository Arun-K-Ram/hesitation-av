import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from "recharts";

// ── WebSocket hook ───────────────────────────────────────────────────────────
function useHesitationStream(url = "ws://localhost:8765") {
  const [data, setData]       = useState(null);
  const [history, setHistory] = useState([]);
  const [status, setStatus]   = useState("connecting");
  const wsRef = useRef(null);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen  = () => setStatus("live");
      ws.onclose = () => { setStatus("disconnected"); setTimeout(connect, 2000); };
      ws.onerror = () => setStatus("error");

      ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        setData(d);
        setHistory(prev => {
          const next = [...prev, {
            t:    d.t,
            A:    d.ambiguity.A,
            Ap:   d.ambiguity.Ap,
            Ab:   d.ambiguity.Ab,
            risk: d.risk.risk,
          }];
          return next.length > 300 ? next.slice(-300) : next;
        });
      };
    };
    connect();
    return () => wsRef.current?.close();
  }, [url]);

  return { data, history, status };
}

// ── Colour helpers ───────────────────────────────────────────────────────────
const STATE_COLORS = {
  CRUISE: "#22c55e", PROBE: "#eab308", HOLD: "#f97316",
  COMMIT: "#3b82f6", ABORT: "#ef4444", YIELD: "#a855f7",
};

const bar = (v, color) => (
  <div style={{ display:"flex", alignItems:"center", gap: 8 }}>
    <div style={{
      flex: 1, height: 6, background: "#1e293b", borderRadius: 3, overflow:"hidden"
    }}>
      <div style={{
        width: `${Math.min(100, v * 100)}%`, height: "100%",
        background: color, borderRadius: 3,
        transition: "width 0.1s linear"
      }}/>
    </div>
    <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:11,
                   color:"#94a3b8", minWidth: 38 }}>
      {v.toFixed(3)}
    </span>
  </div>
);

// ── Sub-components ────────────────────────────────────────────────────────────

function StatusDot({ status }) {
  const colors = { live:"#22c55e", connecting:"#eab308",
                   disconnected:"#ef4444", error:"#ef4444" };
  return (
    <span style={{ display:"inline-flex", alignItems:"center", gap:6 }}>
      <span style={{
        width:8, height:8, borderRadius:"50%",
        background: colors[status] ?? "#64748b",
        boxShadow: status === "live" ? `0 0 8px ${colors.live}` : "none",
      }}/>
      <span style={{ fontSize:11, color:"#64748b", textTransform:"uppercase",
                     letterSpacing:"0.08em" }}>{status}</span>
    </span>
  );
}

function StateCard({ state, color, tInState }) {
  return (
    <div style={{
      background: "#0f172a", border: `1px solid ${color}44`,
      borderRadius: 12, padding: "20px 28px",
      boxShadow: `0 0 24px ${color}22`, textAlign:"center"
    }}>
      <div style={{ fontSize:11, color:"#475569", letterSpacing:"0.12em",
                    marginBottom: 8, textTransform:"uppercase" }}>
        Hesitation State
      </div>
      <div style={{
        fontSize: 36, fontWeight: 700, color,
        fontFamily: "'Space Mono', monospace",
        letterSpacing: "0.04em",
        textShadow: `0 0 20px ${color}88`
      }}>
        {state ?? "-"}
      </div>
      <div style={{ marginTop: 8, fontSize: 12,
                    color:"#475569", fontFamily:"monospace" }}>
        {tInState?.toFixed(2)}s in state
      </div>
    </div>
  );
}

function AmbiguityPanel({ amb }) {
  if (!amb) return null;
  return (
    <div style={{ background:"#0f172a", border:"1px solid #1e293b",
                  borderRadius:12, padding:20 }}>
      <div style={{ fontSize:11, color:"#475569", letterSpacing:"0.12em",
                    marginBottom:16, textTransform:"uppercase" }}>
        Ambiguity Decomposition
      </div>
      <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
        <div>
          <div style={{ fontSize:11, color:"#64748b", marginBottom:4 }}>
            A(t) composite
          </div>
          {bar(amb.A ?? 0, "#818cf8")}
        </div>
        <div>
          <div style={{ fontSize:11, color:"#64748b", marginBottom:4 }}>
            Aₚ perceptual (DetConf)
          </div>
          {bar(amb.Ap ?? 0, "#38bdf8")}
        </div>
        <div>
          <div style={{ fontSize:11, color:"#64748b", marginBottom:4 }}>
            A_b behavioral (MotionEntropy)
          </div>
          {bar(amb.Ab ?? 0, "#fb923c")}
        </div>
        <div style={{ display:"flex", gap:24, marginTop:4 }}>
          <div style={{ fontSize:11, color:"#64748b" }}>
            dA/dt&nbsp;
            <span style={{ color: (amb.dA_dt??0) > 0 ? "#ef4444" : "#22c55e",
                           fontFamily:"monospace" }}>
              {(amb.dA_dt ?? 0) > 0 ? "▲" : "▼"}&nbsp;
              {Math.abs(amb.dA_dt ?? 0).toFixed(4)}
            </span>
          </div>
          <div style={{ fontSize:11, color:"#64748b" }}>
            osc&nbsp;
            <span style={{ color:"#94a3b8", fontFamily:"monospace" }}>
              {(amb.osc ?? 0).toFixed(4)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

function RiskPanel({ risk }) {
  if (!risk) return null;
  return (
    <div style={{ background:"#0f172a", border:"1px solid #1e293b",
                  borderRadius:12, padding:20 }}>
      <div style={{ fontSize:11, color:"#475569", letterSpacing:"0.12em",
                    marginBottom:16, textTransform:"uppercase" }}>
        Risk Composite
      </div>
      <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
        <div>
          <div style={{ fontSize:11, color:"#64748b", marginBottom:4 }}>
            Risk(t) total
          </div>
          {bar(risk.risk ?? 0,
               (risk.risk??0) > 0.7 ? "#ef4444" :
               (risk.risk??0) > 0.45 ? "#f97316" : "#22c55e")}
        </div>
        <div>
          <div style={{ fontSize:11, color:"#64748b", marginBottom:4 }}>
            TTC_risk
          </div>
          {bar(risk.ttc_risk ?? 0, "#f472b6")}
        </div>
        <div>
          <div style={{ fontSize:11, color:"#64748b", marginBottom:4 }}>
            Trajectory Conflict
          </div>
          {bar(risk.traj_conf ?? 0, "#fb923c")}
        </div>
        <div>
          <div style={{ fontSize:11, color:"#64748b", marginBottom:4 }}>
            Correction Severity
          </div>
          {bar(risk.correction ?? 0, "#a78bfa")}
        </div>
      </div>
    </div>
  );
}

function HQMPanel({ hqm }) {
  if (!hqm) return null;
  const macro    = hqm.macro ?? 0;
  const baseline = hqm.greedy_baseline ?? 0.60;
  const delta    = macro - baseline;
  const last     = hqm.last;

  return (
    <div style={{ background:"#0f172a", border:"1px solid #1e293b",
                  borderRadius:12, padding:20 }}>
      <div style={{ fontSize:11, color:"#475569", letterSpacing:"0.12em",
                    marginBottom:16, textTransform:"uppercase" }}>
        Hesitation Quality Metric
      </div>

      <div style={{ display:"flex", justifyContent:"space-between",
                    alignItems:"flex-end", marginBottom:16 }}>
        <div>
          <div style={{ fontSize:11, color:"#64748b" }}>macro HQM</div>
          <div style={{ fontSize:32, fontWeight:700, fontFamily:"monospace",
                        color: delta >= 0 ? "#22c55e" : "#ef4444" }}>
            {macro.toFixed(3)}
          </div>
        </div>
        <div style={{ textAlign:"right" }}>
          <div style={{ fontSize:11, color:"#64748b" }}>vs greedy baseline</div>
          <div style={{ fontSize:18, fontFamily:"monospace",
                        color: delta >= 0 ? "#22c55e" : "#ef4444" }}>
            {delta >= 0 ? "+" : ""}{delta.toFixed(3)}
          </div>
        </div>
      </div>

      <div style={{ background:"#020617", borderRadius:8,
                    padding:"2px 0", marginBottom:12 }}>
        <div style={{
          width:`${Math.min(100, Math.max(0, macro * 100))}%`,
          height:8, borderRadius:8,
          background: delta >= 0 ? "#22c55e" : "#ef4444",
          transition:"width 0.3s ease"
        }}/>
      </div>

      {last && (
        <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr",
                      gap:8, marginTop:8 }}>
          {["S","E","B","R"].map(k => (
            <div key={k} style={{ background:"#020617", borderRadius:8,
                                   padding:"8px 12px" }}>
              <div style={{ fontSize:10, color:"#475569" }}>{k}</div>
              <div style={{ fontFamily:"monospace", fontSize:14,
                            color:"#94a3b8" }}>
                {typeof last[k] === "number" ? last[k].toFixed(3) : last[k]}
              </div>
            </div>
          ))}
        </div>
      )}
      <div style={{ marginTop:10, fontSize:11, color:"#334155" }}>
        {hqm.n_episodes ?? 0} episodes completed
      </div>
    </div>
  );
}

function Timeline({ history }) {
  return (
    <div style={{ background:"#0f172a", border:"1px solid #1e293b",
                  borderRadius:12, padding:20 }}>
      <div style={{ fontSize:11, color:"#475569", letterSpacing:"0.12em",
                    marginBottom:12, textTransform:"uppercase" }}>
        Temporal Ambiguity & Risk (last 10s)
      </div>
      <ResponsiveContainer width="100%" height={140}>
        <AreaChart data={history} margin={{ top:4, right:4, left:-20, bottom:0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="t" tick={{ fill:"#334155", fontSize:9 }}
                 tickFormatter={v => v.toFixed(0) + "s"} />
          <YAxis domain={[0,1]} tick={{ fill:"#334155", fontSize:9 }} />
          <Tooltip
            contentStyle={{ background:"#0f172a", border:"1px solid #1e293b",
                            fontSize:11, color:"#94a3b8" }}
            formatter={(v,n) => [v.toFixed(3), n]}
          />
          <ReferenceLine y={0.35} stroke="#374151" strokeDasharray="4 2"
                         label={{ value:"τ_low", fill:"#374151", fontSize:9 }} />
          <ReferenceLine y={0.65} stroke="#374151" strokeDasharray="4 2"
                         label={{ value:"τ_high", fill:"#374151", fontSize:9 }} />
          <Area type="monotone" dataKey="A"  stroke="#818cf8" fill="#818cf822"
                strokeWidth={2} dot={false} name="A(t)" />
          <Area type="monotone" dataKey="Ap" stroke="#38bdf8" fill="none"
                strokeWidth={1} dot={false} name="Aₚ" strokeDasharray="4 2" />
          <Area type="monotone" dataKey="Ab" stroke="#fb923c" fill="none"
                strokeWidth={1} dot={false} name="Ab" strokeDasharray="4 2" />
          <Line  type="monotone" dataKey="risk" stroke="#ef444488"
                 strokeWidth={1.5} dot={false} name="Risk" />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const { data, history, status } = useHesitationStream();
  const state = data?.state ?? "CRUISE";
  const color = STATE_COLORS[state] ?? "#64748b";

  return (
    <div style={{
      minHeight: "100vh", background: "#020617",
      color: "#e2e8f0", fontFamily: "'DM Sans', sans-serif",
      padding: "24px 32px",
    }}>
      {/* Header */}
      <div style={{ display:"flex", justifyContent:"space-between",
                    alignItems:"center", marginBottom:28 }}>
        <div>
          <div style={{ fontSize:20, fontWeight:700, letterSpacing:"-0.02em" }}>
            Hesitation-AV
          </div>
          <div style={{ fontSize:12, color:"#475569", marginTop:2 }}>
            Ambiguity-Driven Temporal Decision Framework
          </div>
        </div>
        <StatusDot status={status} />
      </div>

      {/* Top row: state + ambiguity + risk */}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr 1fr",
                    gap:16, marginBottom:16 }}>
        <StateCard state={state} color={color} tInState={data?.t_in_state} />
        <AmbiguityPanel amb={data?.ambiguity} />
        <RiskPanel risk={data?.risk} />
      </div>

      {/* Timeline */}
      <div style={{ marginBottom:16 }}>
        <Timeline history={history} />
      </div>

      {/* Bottom row: HQM + flags */}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>
        <HQMPanel hqm={data?.hqm} />

        <div style={{ background:"#0f172a", border:"1px solid #1e293b",
                      borderRadius:12, padding:20 }}>
          <div style={{ fontSize:11, color:"#475569", letterSpacing:"0.12em",
                        marginBottom:16, textTransform:"uppercase" }}>
            System Flags
          </div>
          {data?.flags ? Object.entries(data.flags).map(([k, v]) => (
            <div key={k} style={{ display:"flex", justifyContent:"space-between",
                                   padding:"8px 0",
                                   borderBottom:"1px solid #0f172a" }}>
              <span style={{ fontSize:12, color:"#64748b",
                             fontFamily:"monospace" }}>{k}</span>
              <span style={{ fontSize:12, fontFamily:"monospace",
                             color: v === true ? "#ef4444"
                                  : v === false ? "#22c55e" : "#94a3b8" }}>
                {String(v)}
              </span>
            </div>
          )) : (
            <div style={{ color:"#334155", fontSize:12 }}>
              Waiting for stream…
            </div>
          )}
          {data?.transition && (
            <div style={{ marginTop:16, padding:"8px 12px",
                          background:"#0a1628", borderRadius:8,
                          border:`1px solid ${color}44` }}>
              <span style={{ fontSize:10, color:"#475569" }}>
                Last transition&nbsp;
              </span>
              <span style={{ fontFamily:"monospace", fontSize:13, color }}>
                {data.transition}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
