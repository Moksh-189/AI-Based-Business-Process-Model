import { useState, useEffect, useRef } from 'react';
import { X, Loader2, CheckCircle2, AlertTriangle, Terminal, TrendingUp } from 'lucide-react';

interface TrainingMessage {
    type: 'progress' | 'log' | 'training_log' | 'complete' | 'error';
    text?: string;
    step?: number;
    total?: number;
    pct?: number;
    reward?: number;
    results?: any;
}

interface TrainingModalProps {
    isOpen: boolean;
    onClose: () => void;
}

const TrainingModal = ({ isOpen, onClose }: TrainingModalProps) => {
    const [messages, setMessages] = useState<TrainingMessage[]>([]);
    const [progress, setProgress] = useState(0);
    const [currentStep, setCurrentStep] = useState(0);
    const [totalSteps, setTotalSteps] = useState(200000);
    const [currentReward, setCurrentReward] = useState(0);
    const [rewardHistory, setRewardHistory] = useState<number[]>([]);
    const [status, setStatus] = useState<'connecting' | 'training' | 'complete' | 'error'>('connecting');
    const logRef = useRef<HTMLDivElement>(null);
    const wsRef = useRef<WebSocket | null>(null);

    useEffect(() => {
        if (!isOpen) return;

        // Reset state
        setMessages([]);
        setProgress(0);
        setCurrentStep(0);
        setCurrentReward(0);
        setRewardHistory([]);
        setStatus('connecting');

        // Start training
        const startTraining = async () => {
            try {
                const res = await fetch('http://localhost:8000/api/optimize', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'already_training') {
                    setStatus('training');
                }
            } catch (err) {
                console.error('Failed to start training:', err);
            }
        };
        startTraining();

        // Connect WebSocket
        const connectWs = () => {
            const ws = new WebSocket('ws://localhost:8000/ws/training');
            wsRef.current = ws;

            ws.onopen = () => {
                setStatus('training');
            };

            ws.onmessage = (event) => {
                try {
                    const msg: TrainingMessage = JSON.parse(event.data);
                    setMessages(prev => [...prev.slice(-150), msg]); // Keep last 150 messages

                    if (msg.type === 'progress') {
                        setProgress(msg.pct || 0);
                        setCurrentStep(msg.step || 0);
                        if (msg.total) setTotalSteps(msg.total);
                        if (msg.reward !== undefined) {
                            setCurrentReward(msg.reward);
                            setRewardHistory(prev => [...prev.slice(-50), msg.reward!]);
                        }
                    } else if (msg.type === 'complete') {
                        setStatus('complete');
                        setProgress(100);
                    } else if (msg.type === 'error') {
                        setStatus('error');
                    }
                } catch {
                    // Plain text message
                    setMessages(prev => [...prev.slice(-150), { type: 'log', text: event.data }]);
                }
            };

            ws.onclose = () => {
                if (status !== 'complete') {
                    // Try to reconnect after a brief delay
                    setTimeout(() => {
                        if (isOpen && status === 'training') {
                            connectWs();
                        }
                    }, 2000);
                }
            };

            ws.onerror = () => {
                setStatus('error');
            };
        };

        // Small delay to let training start
        setTimeout(connectWs, 500);

        return () => {
            wsRef.current?.close();
        };
    }, [isOpen]);

    // Auto-scroll log
    useEffect(() => {
        if (logRef.current) {
            logRef.current.scrollTop = logRef.current.scrollHeight;
        }
    }, [messages]);

    if (!isOpen) return null;

    // Mini reward chart — simple bar visualization
    const maxReward = Math.max(...rewardHistory, 1);
    const minReward = Math.min(...rewardHistory, 0);
    const range = maxReward - minReward || 1;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={status === 'complete' || status === 'error' ? onClose : undefined} />

            {/* Modal */}
            <div className="relative w-full max-w-2xl bg-surface/95 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl overflow-hidden">
                {/* Header */}
                <div className="p-5 border-b border-white/10 flex items-center justify-between bg-gradient-to-r from-primary/10 to-transparent">
                    <div className="flex items-center gap-3">
                        {status === 'training' && <Loader2 size={20} className="text-primary animate-spin" />}
                        {status === 'complete' && <CheckCircle2 size={20} className="text-emerald-400" />}
                        {status === 'error' && <AlertTriangle size={20} className="text-red-400" />}
                        {status === 'connecting' && <Loader2 size={20} className="text-gray-400 animate-spin" />}
                        <div>
                            <h2 className="font-bold text-lg text-white">
                                {status === 'connecting' && 'Initializing Training...'}
                                {status === 'training' && 'GNN-Enhanced RL Training'}
                                {status === 'complete' && 'Training Complete!'}
                                {status === 'error' && 'Training Error'}
                            </h2>
                            <p className="text-xs text-gray-400">PPO Agent · GELU Activation · 200K Timesteps</p>
                        </div>
                    </div>
                    {(status === 'complete' || status === 'error') && (
                        <button
                            onClick={onClose}
                            className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
                        >
                            <X size={18} />
                        </button>
                    )}
                </div>

                {/* Progress Bar */}
                <div className="px-5 pt-4">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-mono text-gray-400">
                            {currentStep.toLocaleString()} / {totalSteps.toLocaleString()} timesteps
                        </span>
                        <span className={`text-xs font-bold font-mono ${status === 'complete' ? 'text-emerald-400' : 'text-primary'}`}>
                            {progress.toFixed(1)}%
                        </span>
                    </div>
                    <div className="h-3 bg-white/5 rounded-full overflow-hidden">
                        <div
                            className={`h-full rounded-full transition-all duration-500 ${status === 'complete'
                                    ? 'bg-gradient-to-r from-emerald-500 to-emerald-400'
                                    : 'bg-gradient-to-r from-primary to-cyan-400'
                                }`}
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>

                {/* Stats Row */}
                <div className="px-5 pt-4 flex gap-4">
                    <div className="flex-1 p-3 rounded-xl bg-white/[0.03] border border-white/5">
                        <div className="flex items-center gap-2 text-gray-400 text-[10px] uppercase tracking-wider mb-1">
                            <TrendingUp size={10} /> Avg Reward
                        </div>
                        <div className="text-lg font-bold font-mono text-primary">{currentReward.toFixed(2)}</div>
                    </div>
                    <div className="flex-1 p-3 rounded-xl bg-white/[0.03] border border-white/5">
                        <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Reward Trend</div>
                        {/* Mini sparkline chart */}
                        <div className="flex items-end gap-[2px] h-6">
                            {rewardHistory.slice(-30).map((r, i) => (
                                <div
                                    key={i}
                                    className="flex-1 bg-primary/60 rounded-t-sm transition-all"
                                    style={{ height: `${Math.max(4, ((r - minReward) / range) * 100)}%` }}
                                />
                            ))}
                            {rewardHistory.length === 0 && (
                                <span className="text-xs text-gray-600 italic">Waiting for data...</span>
                            )}
                        </div>
                    </div>
                </div>

                {/* Log Feed */}
                <div className="px-5 pt-4 pb-5">
                    <div className="flex items-center gap-2 mb-2">
                        <Terminal size={12} className="text-gray-500" />
                        <span className="text-[10px] uppercase tracking-wider text-gray-500 font-medium">Live Log</span>
                    </div>
                    <div
                        ref={logRef}
                        className="h-32 overflow-y-auto bg-black/30 rounded-xl p-3 font-mono text-xs text-gray-400 space-y-0.5 scrollbar-thin"
                    >
                        {messages.map((msg, i) => (
                            <div
                                key={i}
                                className={`
                                    ${msg.type === 'progress' ? 'text-primary' : ''}
                                    ${msg.type === 'complete' ? 'text-emerald-400 font-bold' : ''}
                                    ${msg.type === 'error' ? 'text-red-400' : ''}
                                `}
                            >
                                {msg.type === 'progress'
                                    ? `[${msg.pct?.toFixed(1)}%] Step ${msg.step?.toLocaleString()} | Reward: ${msg.reward?.toFixed(2)}`
                                    : msg.text || JSON.stringify(msg)
                                }
                            </div>
                        ))}
                        {messages.length === 0 && (
                            <div className="text-gray-600 italic">Waiting for training output...</div>
                        )}
                    </div>
                </div>

                {/* Complete Banner */}
                {status === 'complete' && (
                    <div className="px-5 pb-5">
                        <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center gap-3">
                            <CheckCircle2 size={20} className="text-emerald-400 shrink-0" />
                            <div className="flex-1">
                                <p className="text-sm font-bold text-emerald-300">Training Complete</p>
                                <p className="text-xs text-gray-400">Model saved to ppo_gnn_best.zip. Telemetry will update automatically.</p>
                            </div>
                            <button
                                onClick={onClose}
                                className="px-4 py-2 bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-400 rounded-lg text-sm font-bold transition-colors"
                            >
                                Done
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default TrainingModal;
