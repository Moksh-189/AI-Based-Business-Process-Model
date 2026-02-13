import { useEffect, useState } from 'react';
import { useToast } from '../context/ToastContext';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer
} from 'recharts';
import { Activity, Zap } from 'lucide-react';

const initialData = [
    { name: 'Cycle Time (Days)', Baseline: 45, Optimized: 0, amt: 2400 },
    { name: 'Throughput (/mo)', Baseline: 120, Optimized: 0, amt: 2210 },
    { name: 'OpEx ($k/mo)', Baseline: 85, Optimized: 0, amt: 2290 },
];

const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-surface border border-white/10 p-3 rounded-lg shadow-xl backdrop-blur-md">
                <p className="font-bold text-gray-200 mb-2">{label}</p>
                <p className="text-primary text-sm">
                    Baseline: <span className="font-mono">{payload[0].value}</span>
                </p>
                <p className="text-emerald-400 text-sm">
                    Optimized: <span className="font-mono">{payload[1].value}</span>
                </p>
            </div>
        );
    }
    return null;
};

const TelemetryPanel = () => {
    const { showToast } = useToast();
    console.log("TelemetryPanel rendering");
    const [data, setData] = useState(initialData);

    useEffect(() => {
        // Test toast on mount
        showToast("Telemetry Panel Mounted", "info", 5000);

        const fetchData = async () => {
            try {
                const response = await fetch('http://localhost:8000/api/telemetry');
                if (response.ok) {
                    const jsonData = await response.json();
                    setData(jsonData);
                }
            } catch (error) {
                console.error("Error fetching telemetry:", error);
            }
        };

        // Initial fetch
        fetchData();

        // Poll every 2 seconds
        const interval = setInterval(fetchData, 2000);
        return () => clearInterval(interval);
    }, []);





    const handleOptimize = async () => {
        showToast("Optimization agent starting...", "loading", 2000);
        try {
            await fetch('http://localhost:8000/api/optimize', { method: 'POST' });
            // Since it's async background process, we just notify start.
            // Ideally we polll for status, but for now:
            setTimeout(() => {
                showToast("Optimization process running on backend.", "info");
            }, 1000);
        } catch (error) {
            console.error("Optimization failed:", error);
            showToast("Failed to start optimization.", "error");
        }
    };

    return (
        <div className="h-[300px] w-full bg-surface/30 backdrop-blur-sm border border-white/10 rounded-2xl overflow-hidden flex flex-col">
            <div className="p-4 border-b border-white/10 bg-primary/5 flex justify-between items-center relative z-50">
                <h3 className="font-bold text-lg text-primary flex items-center gap-2">
                    <Activity size={20} /> Live Telemetry
                </h3>
                <div className="flex items-center gap-3">
                    <button
                        data-testid="auto-optimize-btn-v2"
                        onClick={handleOptimize}
                        className="flex items-center gap-2 px-3 py-1.5 bg-amber-500/10 hover:bg-amber-500/20 text-amber-500 border border-amber-500/50 rounded-lg text-xs font-bold transition-all hover:scale-105 active:scale-95"
                    >
                        <Zap size={14} /> AUTO-OPTIMIZE
                    </button>
                    <span className="text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/30 px-2 py-1 rounded">
                        ROI +24% Proj.
                    </span>
                </div>
            </div>

            <div className="flex-1 p-4 min-h-0">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={data}
                        margin={{
                            top: 5,
                            right: 30,
                            left: 20,
                            bottom: 5,
                        }}
                    >
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                        <XAxis dataKey="name" tick={{ fill: '#94a3b8' }} axisLine={{ stroke: '#334155' }} tickLine={false} />
                        <YAxis tick={{ fill: '#94a3b8' }} axisLine={{ stroke: '#334155' }} tickLine={false} />
                        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                        <Legend wrapperStyle={{ paddingTop: '10px' }} />
                        <Bar dataKey="Baseline" fill="#64748b" radius={[4, 4, 0, 0]} animationDuration={1000} />
                        <Bar dataKey="Optimized" fill="#06b6d4" radius={[4, 4, 0, 0]} animationDuration={1000} />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default TelemetryPanel;
