import { useState, useEffect, useCallback, useRef } from 'react';
import { API_URL } from '../config/api';
import { useToast } from '../context/ToastContext';
import { DragDropContext, Droppable, Draggable, type DropResult } from '@hello-pangea/dnd';
import { Users, User, AlertCircle, Sparkles, X, TrendingDown, TrendingUp, DollarSign, Loader2, Network, Play, CheckCircle2, Zap, BarChart3 } from 'lucide-react';

interface Employee {
    id: string;
    name: string;
    role: string;
    efficiency: number;
}

interface ProcessNode {
    id: string;
    data: {
        label: string;
        isBottleneck: boolean;
        avgDuration: number;
    };
}

interface Simulation {
    cycle_time_before: number;
    cycle_time_after: number;
    cycle_reduction_pct: number;
    throughput_gain_pct: number;
    opex_increase: number;
    is_bottleneck: boolean;
    impact_score: number;
}

interface Suggestion {
    simulation: Simulation;
    ai_suggestion: string;
}

const initialEmployees: Employee[] = [
    { id: 'emp-1', name: 'Alice Chen', role: 'Snr. Analyst', efficiency: 95 },
    { id: 'emp-2', name: 'Bob Smith', role: 'Approver', efficiency: 78 },
    { id: 'emp-3', name: 'Charlie D', role: 'Clerk', efficiency: 82 },
    { id: 'emp-4', name: 'Dana Lee', role: 'QA Engineer', efficiency: 88 },
    { id: 'emp-5', name: 'Eve Park', role: 'Snr. Approver', efficiency: 91 },
    { id: 'emp-6', name: 'Frank Wu', role: 'Analyst', efficiency: 75 },
];

const WorkforcePage = () => {
    const { showToast } = useToast();
    const [processes, setProcesses] = useState<ProcessNode[]>([]);
    const [availableStaff, setAvailableStaff] = useState<Employee[]>(initialEmployees);
    const [assignments, setAssignments] = useState<Record<string, Employee[]>>({});
    const [suggestion, setSuggestion] = useState<Suggestion | null>(null);
    const [suggestionProcess, setSuggestionProcess] = useState<string>('');
    const [isLoadingSuggestion, setIsLoadingSuggestion] = useState(false);
    const [showSuggestion, setShowSuggestion] = useState(false);
    const [showSimModal, setShowSimModal] = useState(false);
    const [simPhase, setSimPhase] = useState<'idle' | 'simulating' | 'analyzing' | 'complete'>('idle');
    const [simProgress, setSimProgress] = useState<{ current: number; total: number; label: string }>({ current: 0, total: 0, label: '' });
    const [simResults, setSimResults] = useState<{ processName: string; simulation: Simulation; suggestion: string }[]>([]);
    const [simAggregate, setSimAggregate] = useState<Simulation | null>(null);
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const poolContainerRef = useRef<HTMLDivElement>(null);
    const isDraggingRef = useRef(false);
    const scrollAnimRef = useRef<number | null>(null);

    // Auto-scroll during drag when cursor is near edges
    useEffect(() => {
        const EDGE_SIZE = 80; // px from edge to start scrolling
        const SCROLL_SPEED = 8; // px per frame

        const handleMouseMove = (e: MouseEvent) => {
            if (!isDraggingRef.current) return;

            // Auto-scroll the process cards (left column)
            const scrollEl = scrollContainerRef.current;
            if (scrollEl) {
                const rect = scrollEl.getBoundingClientRect();
                const y = e.clientY;

                if (y < rect.top + EDGE_SIZE && y > rect.top) {
                    // Near top edge — scroll up
                    const intensity = 1 - (y - rect.top) / EDGE_SIZE;
                    scrollEl.scrollTop -= SCROLL_SPEED * intensity;
                } else if (y > rect.bottom - EDGE_SIZE && y < rect.bottom) {
                    // Near bottom edge — scroll down
                    const intensity = 1 - (rect.bottom - y) / EDGE_SIZE;
                    scrollEl.scrollTop += SCROLL_SPEED * intensity;
                }
            }

            // Auto-scroll the employee pool (right column)
            const poolEl = poolContainerRef.current;
            if (poolEl) {
                const rect = poolEl.getBoundingClientRect();
                const y = e.clientY;

                if (y < rect.top + EDGE_SIZE && y > rect.top) {
                    const intensity = 1 - (y - rect.top) / EDGE_SIZE;
                    poolEl.scrollTop -= SCROLL_SPEED * intensity;
                } else if (y > rect.bottom - EDGE_SIZE && y < rect.bottom) {
                    const intensity = 1 - (rect.bottom - y) / EDGE_SIZE;
                    poolEl.scrollTop += SCROLL_SPEED * intensity;
                }
            }
        };

        // Use requestAnimationFrame for smooth continuous scrolling
        let lastMouseEvent: MouseEvent | null = null;

        const onMouseMove = (e: MouseEvent) => {
            lastMouseEvent = e;
        };

        const tick = () => {
            if (lastMouseEvent && isDraggingRef.current) {
                handleMouseMove(lastMouseEvent);
            }
            scrollAnimRef.current = requestAnimationFrame(tick);
        };

        window.addEventListener('mousemove', onMouseMove);
        scrollAnimRef.current = requestAnimationFrame(tick);

        return () => {
            window.removeEventListener('mousemove', onMouseMove);
            if (scrollAnimRef.current) cancelAnimationFrame(scrollAnimRef.current);
        };
    }, []);

    // Fetch process topology
    useEffect(() => {
        const fetchTopology = async () => {
            try {
                const res = await fetch(`${API_URL}/api/topology`);
                const data = await res.json();
                setProcesses(data.nodes || []);
                // Initialize assignments map
                const initialAssignments: Record<string, Employee[]> = {};
                data.nodes.forEach((n: ProcessNode) => {
                    initialAssignments[n.id] = [];
                });
                setAssignments(initialAssignments);
            } catch (err) {
                console.error('Failed to fetch topology:', err);
                showToast('Failed to load process data', 'error');
            }
        };
        fetchTopology();
    }, []);

    // Fetch AI suggestion after assignment
    const fetchSuggestion = useCallback(async (processId: string, processLabel: string, assigned: Employee[]) => {
        setIsLoadingSuggestion(true);
        setSuggestionProcess(processLabel);
        setShowSuggestion(true);

        try {
            const res = await fetch(`${API_URL}/api/suggest`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    process_id: processId,
                    process_label: processLabel,
                    assigned: assigned,
                }),
            });
            const data = await res.json();
            setSuggestion(data);
        } catch (err) {
            console.error('Suggestion fetch failed:', err);
            showToast('Could not get AI suggestion', 'error');
        } finally {
            setIsLoadingSuggestion(false);
        }
    }, [showToast]);

    // No auto-hide: User must manually close the suggestion panel

    const onDragStart = () => {
        isDraggingRef.current = true;
    };

    const onDragEnd = (result: DropResult) => {
        isDraggingRef.current = false;
        if (!result.destination) return;
        const { source, destination } = result;
        if (source.droppableId === destination.droppableId) return;

        // Find the dragged employee
        let draggedEmployee: Employee | null = null;

        if (source.droppableId === 'employee-pool') {
            // Dragging from pool
            draggedEmployee = availableStaff[source.index];
            if (!draggedEmployee) return;

            // Check if destination is a process (starts with 'process-')
            if (destination.droppableId.startsWith('process-')) {
                const processId = destination.droppableId.replace('process-', '');

                // Remove from pool
                const newPool = [...availableStaff];
                newPool.splice(source.index, 1);
                setAvailableStaff(newPool);

                // Add to process
                const newAssignments = { ...assignments };
                newAssignments[processId] = [...(newAssignments[processId] || []), draggedEmployee];
                setAssignments(newAssignments);

                // Find process label
                const proc = processes.find(p => p.id === processId);
                showToast(`Assigned ${draggedEmployee.name} to ${proc?.data.label || 'process'}`, 'success', 3000);

                // Trigger AI suggestion
                if (proc) {
                    fetchSuggestion(processId, proc.data.label, newAssignments[processId]);
                }
            }
        } else if (source.droppableId.startsWith('process-')) {
            // Dragging from a process back to the pool
            const processId = source.droppableId.replace('process-', '');
            const processStaff = assignments[processId] || [];
            draggedEmployee = processStaff[source.index];
            if (!draggedEmployee) return;

            if (destination.droppableId === 'employee-pool') {
                // Remove from process
                const newProcessStaff = [...processStaff];
                newProcessStaff.splice(source.index, 1);
                const newAssignments = { ...assignments };
                newAssignments[processId] = newProcessStaff;
                setAssignments(newAssignments);

                // Add back to pool
                setAvailableStaff(prev => [...prev, draggedEmployee!]);
                showToast(`${draggedEmployee.name} returned to pool`, 'info', 2000);
            }
        }
    };

    const removeFromProcess = (processId: string, empIndex: number) => {
        const processStaff = assignments[processId] || [];
        const emp = processStaff[empIndex];
        if (!emp) return;

        const newStaff = [...processStaff];
        newStaff.splice(empIndex, 1);
        setAssignments(prev => ({ ...prev, [processId]: newStaff }));
        setAvailableStaff(prev => [...prev, emp]);
        showToast(`${emp.name} returned to pool`, 'info', 2000);
    };

    // Check if any process has employees assigned
    const totalAssigned = Object.values(assignments).reduce((sum, arr) => sum + arr.length, 0);

    // Run full simulation for all assignments — with modal phases
    const runFullSimulation = useCallback(async () => {
        const assignedProcesses = Object.entries(assignments).filter(([, staff]) => staff.length > 0);
        if (assignedProcesses.length === 0) {
            showToast('No employees assigned yet', 'error', 3000);
            return;
        }

        // Open modal and start simulation
        setShowSimModal(true);
        setSimPhase('simulating');
        setSimResults([]);
        setSimAggregate(null);

        try {
            const allResults: { processName: string; simulation: Simulation; suggestion: string }[] = [];

            // Phase 1: Simulate each process one-by-one with visible progress
            for (let i = 0; i < assignedProcesses.length; i++) {
                const [processId, staff] = assignedProcesses[i];
                const proc = processes.find(p => p.id === processId);
                const label = proc?.data.label || 'Unknown';

                setSimProgress({ current: i + 1, total: assignedProcesses.length, label });

                // Minimum 600ms per process so user can see each step
                const [res] = await Promise.all([
                    fetch(`${API_URL}/api/suggest`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ process_id: processId, process_label: label, assigned: staff }),
                    }).then(r => r.json()),
                    new Promise(resolve => setTimeout(resolve, 600)),
                ]);

                allResults.push({ processName: label, simulation: res.simulation, suggestion: res.ai_suggestion });
                setSimResults([...allResults]);
            }

            // Phase 2: Analyzing
            setSimPhase('analyzing');
            await new Promise(resolve => setTimeout(resolve, 1200));

            // Aggregate
            const agg: Simulation = {
                cycle_time_before: allResults.reduce((s, r) => s + r.simulation.cycle_time_before, 0) / allResults.length,
                cycle_time_after: allResults.reduce((s, r) => s + r.simulation.cycle_time_after, 0) / allResults.length,
                cycle_reduction_pct: allResults.reduce((s, r) => s + r.simulation.cycle_reduction_pct, 0) / allResults.length,
                throughput_gain_pct: allResults.reduce((s, r) => s + r.simulation.throughput_gain_pct, 0) / allResults.length,
                opex_increase: allResults.reduce((s, r) => s + r.simulation.opex_increase, 0),
                is_bottleneck: allResults.some(r => r.simulation.is_bottleneck),
                impact_score: allResults.reduce((s, r) => s + r.simulation.impact_score, 0) / allResults.length,
            };
            setSimAggregate(agg);

            // Phase 3: Complete
            setSimPhase('complete');
        } catch (err) {
            console.error('Full simulation failed:', err);
            showToast('Simulation failed', 'error');
            setShowSimModal(false);
            setSimPhase('idle');
        }
    }, [assignments, processes, showToast]);

    return (
        <div className="h-full flex flex-col gap-6">
            {/* Page Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-emerald-400">
                        Workforce Allocation
                    </h1>
                    <p className="text-gray-400 text-sm mt-1">Drag employees onto process cards to assign & simulate</p>
                </div>
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-primary/10 border border-primary/20 text-primary text-xs font-medium">
                        <Network size={14} />
                        {processes.length} Processes · {availableStaff.length} Available
                    </div>
                    {totalAssigned > 0 && (
                        <button
                            onClick={runFullSimulation}
                            disabled={isLoadingSuggestion}
                            className="flex items-center gap-2 px-4 py-2 rounded-xl bg-gradient-to-r from-primary to-emerald-500 text-white text-sm font-bold shadow-lg shadow-primary/25 hover:shadow-primary/40 hover:scale-105 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            {isLoadingSuggestion ? (
                                <Loader2 size={16} className="animate-spin" />
                            ) : (
                                <Play size={16} fill="currentColor" />
                            )}
                            Run Digital Twin
                        </button>
                    )}
                </div>
            </div>

            <DragDropContext onDragStart={onDragStart} onDragEnd={onDragEnd}>
                <div className="flex-1 flex gap-6 min-h-0 overflow-hidden">

                    {/* LEFT: Process Cards (Drop Targets) */}
                    <div ref={scrollContainerRef} className="flex-[3] overflow-y-auto pr-2 space-y-4 scrollbar-thin">
                        <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-2 sticky top-0 bg-background/80 backdrop-blur-sm py-2 z-10">
                            Process Steps — Drop Employees Here
                        </h3>
                        {processes.map((proc) => {
                            const isBottleneck = proc.data.isBottleneck;
                            const assigned = assignments[proc.id] || [];

                            return (
                                <Droppable key={proc.id} droppableId={`process-${proc.id}`}>
                                    {(provided, snapshot) => (
                                        <div
                                            ref={provided.innerRef}
                                            {...provided.droppableProps}
                                            className={`
                                                p-4 rounded-xl border-2 transition-all duration-300
                                                ${isBottleneck
                                                    ? 'bg-red-950/30 border-red-500/40 shadow-[0_0_20px_rgba(239,68,68,0.15)]'
                                                    : 'bg-surface/40 border-white/10'
                                                }
                                                ${snapshot.isDraggingOver
                                                    ? isBottleneck
                                                        ? 'border-red-400 bg-red-950/50 scale-[1.01] shadow-[0_0_30px_rgba(239,68,68,0.3)]'
                                                        : 'border-primary/60 bg-primary/10 scale-[1.01] shadow-[0_0_20px_rgba(6,182,212,0.2)]'
                                                    : ''
                                                }
                                            `}
                                        >
                                            {/* Process Header */}
                                            <div className="flex items-center justify-between mb-3">
                                                <div className="flex items-center gap-3">
                                                    {isBottleneck && (
                                                        <div className="p-1.5 rounded-md bg-red-500/20 animate-pulse">
                                                            <AlertCircle size={14} className="text-red-400" />
                                                        </div>
                                                    )}
                                                    <div>
                                                        <h4 className={`font-bold text-sm ${isBottleneck ? 'text-red-300' : 'text-white'}`}>
                                                            {proc.data.label}
                                                        </h4>
                                                        <span className="text-xs text-gray-500">
                                                            Avg. {proc.data.avgDuration}d
                                                        </span>
                                                    </div>
                                                </div>
                                                <div className="flex items-center gap-2">
                                                    {isBottleneck && (
                                                        <span className="text-[10px] px-2 py-0.5 bg-red-500/20 text-red-400 rounded-full font-bold uppercase tracking-wider">
                                                            Bottleneck
                                                        </span>
                                                    )}
                                                    <span className="text-[10px] px-2 py-0.5 bg-white/5 text-gray-400 rounded-full">
                                                        {assigned.length} assigned
                                                    </span>
                                                </div>
                                            </div>

                                            {/* Assigned Employee Chips */}
                                            <div className="flex flex-wrap gap-2 min-h-[44px]">
                                                {assigned.map((emp, idx) => (
                                                    <Draggable key={emp.id} draggableId={emp.id} index={idx}>
                                                        {(dragProvided) => (
                                                            <div
                                                                ref={dragProvided.innerRef}
                                                                {...dragProvided.draggableProps}
                                                                {...dragProvided.dragHandleProps}
                                                                className="flex items-center gap-2 pl-2 pr-1 py-1 bg-surface/80 border border-white/10 rounded-lg text-xs group hover:border-primary/40 transition-all cursor-grab active:cursor-grabbing"
                                                            >
                                                                <div className="w-5 h-5 rounded-full bg-primary/20 flex items-center justify-center text-primary">
                                                                    <User size={10} />
                                                                </div>
                                                                <span className="font-medium text-white">{emp.name}</span>
                                                                <span className="text-emerald-400 font-mono">{emp.efficiency}%</span>
                                                                <button
                                                                    onClick={(e) => { e.stopPropagation(); removeFromProcess(proc.id, idx); }}
                                                                    className="p-0.5 rounded hover:bg-red-500/20 text-gray-500 hover:text-red-400 transition-colors ml-1"
                                                                >
                                                                    <X size={12} />
                                                                </button>
                                                            </div>
                                                        )}
                                                    </Draggable>
                                                ))}
                                                {assigned.length === 0 && !snapshot.isDraggingOver && (
                                                    <div className="w-full flex items-center justify-center text-gray-600 text-xs italic py-2 border border-dashed border-white/5 rounded-lg">
                                                        Drop employees here
                                                    </div>
                                                )}
                                                {snapshot.isDraggingOver && (
                                                    <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border-2 border-dashed ${isBottleneck ? 'border-red-400/50 text-red-300' : 'border-primary/50 text-primary'} text-xs animate-pulse`}>
                                                        <User size={12} /> Release to assign
                                                    </div>
                                                )}
                                            </div>
                                            {provided.placeholder}
                                        </div>
                                    )}
                                </Droppable>
                            );
                        })}
                    </div>

                    {/* RIGHT: Employee Pool */}
                    <div className="flex-[1] flex flex-col min-h-0">
                        <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                            <Users size={14} /> Employee Pool
                        </h3>
                        <Droppable droppableId="employee-pool">
                            {(provided, snapshot) => (
                                <div
                                    ref={(el) => { (poolContainerRef as any).current = el; provided.innerRef(el); }}
                                    {...provided.droppableProps}
                                    className={`
                                        flex-1 overflow-y-auto space-y-2 p-3 rounded-xl border-2 border-dashed transition-all scrollbar-thin
                                        ${snapshot.isDraggingOver
                                            ? 'border-primary/50 bg-primary/5'
                                            : 'border-white/10 bg-surface/20'
                                        }
                                    `}
                                >
                                    {availableStaff.map((emp, index) => (
                                        <Draggable key={emp.id} draggableId={emp.id} index={index}>
                                            {(dragProvided, dragSnapshot) => (
                                                <div
                                                    ref={dragProvided.innerRef}
                                                    {...dragProvided.draggableProps}
                                                    {...dragProvided.dragHandleProps}
                                                    className={`
                                                        p-3 rounded-xl border transition-all cursor-grab active:cursor-grabbing
                                                        ${dragSnapshot.isDragging
                                                            ? 'bg-primary/20 border-primary/50 shadow-xl shadow-primary/10 scale-105 rotate-2'
                                                            : 'bg-surface/50 border-white/5 hover:bg-surface/80 hover:border-white/20'
                                                        }
                                                    `}
                                                >
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-primary/30 to-accent/20 flex items-center justify-center text-primary border border-primary/20">
                                                            <User size={16} />
                                                        </div>
                                                        <div className="flex-1 min-w-0">
                                                            <div className="font-bold text-sm text-white truncate">{emp.name}</div>
                                                            <div className="text-xs text-gray-500">{emp.role}</div>
                                                        </div>
                                                    </div>
                                                    {/* Efficiency Bar */}
                                                    <div className="mt-2 flex items-center gap-2">
                                                        <div className="flex-1 h-1.5 bg-white/5 rounded-full overflow-hidden">
                                                            <div
                                                                className="h-full rounded-full bg-gradient-to-r from-primary to-emerald-400 transition-all"
                                                                style={{ width: `${emp.efficiency}%` }}
                                                            />
                                                        </div>
                                                        <span className="text-[10px] font-mono text-emerald-400">{emp.efficiency}%</span>
                                                    </div>
                                                </div>
                                            )}
                                        </Draggable>
                                    ))}
                                    {availableStaff.length === 0 && (
                                        <div className="flex flex-col items-center justify-center py-8 text-gray-600 text-xs">
                                            <Users size={24} className="mb-2 opacity-50" />
                                            All staff assigned
                                        </div>
                                    )}
                                    {provided.placeholder}
                                </div>
                            )}
                        </Droppable>
                    </div>
                </div>
            </DragDropContext>

            {/* AI SUGGESTION PANEL (slides up from bottom) */}
            <div className={`
                transition-all duration-500 ease-out overflow-hidden
                ${showSuggestion ? 'max-h-[300px] opacity-100' : 'max-h-0 opacity-0'}
            `}>
                <div className="bg-surface/60 backdrop-blur-xl border border-white/10 rounded-2xl p-5 shadow-2xl shadow-primary/5">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 border border-primary/20">
                                <Sparkles size={18} className="text-primary" />
                            </div>
                            <div>
                                <h3 className="font-bold text-sm text-white">AI Suggestion</h3>
                                <p className="text-xs text-gray-400">
                                    Simulation result for <span className="text-primary">{suggestionProcess}</span>
                                </p>
                            </div>
                        </div>
                        <button
                            onClick={() => setShowSuggestion(false)}
                            className="p-1.5 rounded-lg hover:bg-white/5 text-gray-500 hover:text-white transition-colors"
                        >
                            <X size={16} />
                        </button>
                    </div>

                    {isLoadingSuggestion ? (
                        <div className="flex items-center justify-center py-6 gap-3 text-gray-400">
                            <Loader2 size={20} className="animate-spin text-primary" />
                            <span className="text-sm">Running Digital Twin simulation...</span>
                        </div>
                    ) : suggestion ? (
                        <div className="flex gap-6">
                            {/* Left: Simulation Metrics */}
                            <div className="flex-1 grid grid-cols-3 gap-3">
                                <MetricCard
                                    icon={<TrendingDown size={14} />}
                                    label="Cycle Time"
                                    before={`${suggestion.simulation.cycle_time_before}d`}
                                    after={`${suggestion.simulation.cycle_time_after}d`}
                                    change={`-${suggestion.simulation.cycle_reduction_pct}%`}
                                    positive={true}
                                />
                                <MetricCard
                                    icon={<TrendingUp size={14} />}
                                    label="Throughput"
                                    before="baseline"
                                    after={`+${suggestion.simulation.throughput_gain_pct}%`}
                                    change={`+${suggestion.simulation.throughput_gain_pct}%`}
                                    positive={true}
                                />
                                <MetricCard
                                    icon={<DollarSign size={14} />}
                                    label="OpEx Impact"
                                    before="baseline"
                                    after={`+$${suggestion.simulation.opex_increase}k`}
                                    change={`+$${suggestion.simulation.opex_increase}k`}
                                    positive={false}
                                />
                            </div>

                            {/* Right: AI Text */}
                            <div className="flex-1 p-4 bg-white/[0.02] rounded-xl border border-white/5">
                                <div className="flex items-center gap-2 mb-2">
                                    <div className="w-6 h-6 rounded-full bg-primary/20 flex items-center justify-center">
                                        <Sparkles size={12} className="text-primary" />
                                    </div>
                                    <span className="text-xs font-bold text-primary">AI Recommendation</span>
                                </div>
                                <p className="text-sm text-gray-300 leading-relaxed">
                                    {suggestion.ai_suggestion}
                                </p>
                            </div>
                        </div>
                    ) : null}
                </div>
            </div>

            {/* DIGITAL TWIN SIMULATION MODAL */}
            {showSimModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
                    <div className="w-[680px] max-h-[85vh] bg-surface/95 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col">

                        {/* Modal Header */}
                        <div className="p-5 border-b border-white/10 flex items-center justify-between">
                            <div className="flex items-center gap-3">
                                <div className={`p-2.5 rounded-xl ${simPhase === 'complete' ? 'bg-emerald-500/20' : 'bg-primary/20'} transition-colors`}>
                                    {simPhase === 'complete' ? (
                                        <CheckCircle2 size={20} className="text-emerald-400" />
                                    ) : simPhase === 'analyzing' ? (
                                        <BarChart3 size={20} className="text-amber-400 animate-pulse" />
                                    ) : (
                                        <Zap size={20} className="text-primary animate-pulse" />
                                    )}
                                </div>
                                <div>
                                    <h2 className="font-bold text-lg text-white">Digital Twin Simulation</h2>
                                    <p className="text-xs text-gray-400">
                                        {simPhase === 'simulating' && `Simulating process ${simProgress.current}/${simProgress.total}...`}
                                        {simPhase === 'analyzing' && 'Analyzing aggregate impact...'}
                                        {simPhase === 'complete' && 'Simulation complete — results ready'}
                                    </p>
                                </div>
                            </div>
                            {simPhase === 'complete' && (
                                <button
                                    onClick={() => { setShowSimModal(false); setSimPhase('idle'); }}
                                    className="p-2 rounded-lg hover:bg-white/5 text-gray-500 hover:text-white transition-colors"
                                >
                                    <X size={18} />
                                </button>
                            )}
                        </div>

                        {/* Progress Bar */}
                        {simPhase !== 'complete' && (
                            <div className="px-5 pt-4">
                                <div className="flex items-center justify-between text-xs mb-2">
                                    <span className="text-gray-400 font-medium">
                                        {simPhase === 'simulating' ? simProgress.label : 'Aggregating results'}
                                    </span>
                                    <span className="text-primary font-mono">
                                        {simPhase === 'simulating'
                                            ? `${Math.round((simProgress.current / simProgress.total) * 100)}%`
                                            : '...'
                                        }
                                    </span>
                                </div>
                                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                                    <div
                                        className={`h-full rounded-full transition-all duration-500 ease-out ${simPhase === 'analyzing'
                                            ? 'bg-amber-500 animate-pulse w-full'
                                            : 'bg-gradient-to-r from-primary to-emerald-400'
                                            }`}
                                        style={simPhase === 'simulating' ? {
                                            width: `${(simProgress.current / simProgress.total) * 100}%`
                                        } : undefined}
                                    />
                                </div>
                            </div>
                        )}

                        {/* Live Process Checklist (during simulation) */}
                        {simPhase === 'simulating' && simResults.length > 0 && (
                            <div className="px-5 pt-3 space-y-1.5 max-h-[200px] overflow-y-auto">
                                {simResults.map((r, i) => (
                                    <div key={i} className="flex items-center gap-2 text-xs">
                                        <CheckCircle2 size={14} className="text-emerald-400 shrink-0" />
                                        <span className="text-gray-300">{r.processName}</span>
                                        <span className="text-emerald-400 font-mono ml-auto">-{r.simulation.cycle_reduction_pct.toFixed(0)}% cycle</span>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Analyzing Animation */}
                        {simPhase === 'analyzing' && (
                            <div className="flex items-center justify-center py-12 gap-3 text-gray-400">
                                <Loader2 size={24} className="animate-spin text-amber-400" />
                                <span className="text-sm">Computing aggregate metrics...</span>
                            </div>
                        )}

                        {/* COMPLETE: Results View */}
                        {simPhase === 'complete' && simAggregate && (
                            <div className="flex-1 overflow-y-auto p-5 space-y-5">
                                {/* Aggregate Metrics */}
                                <div className="grid grid-cols-3 gap-3">
                                    <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-center">
                                        <TrendingDown size={18} className="text-emerald-400 mx-auto mb-1" />
                                        <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Cycle Time</div>
                                        <div className="text-2xl font-bold text-emerald-400">-{simAggregate.cycle_reduction_pct.toFixed(0)}%</div>
                                        <div className="text-xs text-gray-500">{simAggregate.cycle_time_before.toFixed(1)}d → {simAggregate.cycle_time_after.toFixed(1)}d</div>
                                    </div>
                                    <div className="p-4 rounded-xl bg-blue-500/10 border border-blue-500/20 text-center">
                                        <TrendingUp size={18} className="text-blue-400 mx-auto mb-1" />
                                        <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">Throughput</div>
                                        <div className="text-2xl font-bold text-blue-400">+{simAggregate.throughput_gain_pct.toFixed(0)}%</div>
                                        <div className="text-xs text-gray-500">Increase per month</div>
                                    </div>
                                    <div className="p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-center">
                                        <DollarSign size={18} className="text-amber-400 mx-auto mb-1" />
                                        <div className="text-[10px] uppercase tracking-wider text-gray-400 mb-1">OpEx Impact</div>
                                        <div className="text-2xl font-bold text-amber-400">+${simAggregate.opex_increase.toFixed(1)}k</div>
                                        <div className="text-xs text-gray-500">Monthly cost change</div>
                                    </div>
                                </div>

                                {/* Per-Process Breakdown */}
                                <div>
                                    <h4 className="text-xs font-bold text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                                        <BarChart3 size={12} /> Per-Process Breakdown
                                    </h4>
                                    <div className="space-y-2">
                                        {simResults.map((r, i) => (
                                            <div key={i} className="p-3 bg-white/[0.03] rounded-xl border border-white/5 flex items-center gap-4">
                                                <div className="flex-1 min-w-0">
                                                    <div className="font-bold text-sm text-white">{r.processName}</div>
                                                    <div className="text-xs text-gray-500 truncate mt-0.5">{r.suggestion}</div>
                                                </div>
                                                <div className="flex items-center gap-3 shrink-0 text-xs font-mono">
                                                    <span className="text-emerald-400">-{r.simulation.cycle_reduction_pct.toFixed(0)}%</span>
                                                    <span className="text-blue-400">+{r.simulation.throughput_gain_pct.toFixed(0)}%</span>
                                                    <span className="text-amber-400">+${r.simulation.opex_increase.toFixed(1)}k</span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Close Button */}
                                <button
                                    onClick={() => { setShowSimModal(false); setSimPhase('idle'); }}
                                    className="w-full py-3 rounded-xl bg-gradient-to-r from-primary to-emerald-500 text-white font-bold text-sm hover:shadow-lg hover:shadow-primary/25 transition-all"
                                >
                                    Apply & Close
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

// Metric Card Subcomponent
const MetricCard = ({ icon, label, before, after, change, positive }: {
    icon: React.ReactNode;
    label: string;
    before: string;
    after: string;
    change: string;
    positive: boolean;
}) => (
    <div className="p-3 rounded-xl bg-white/[0.03] border border-white/5 space-y-2">
        <div className="flex items-center gap-2 text-gray-400">
            {icon}
            <span className="text-[10px] font-medium uppercase tracking-wider">{label}</span>
        </div>
        <div className="flex items-baseline justify-between">
            <span className="text-xs text-gray-500 line-through">{before}</span>
            <span className="text-sm font-bold text-white">{after}</span>
        </div>
        <div className={`text-xs font-mono font-bold ${positive ? 'text-emerald-400' : 'text-amber-400'}`}>
            {change}
        </div>
    </div>
);

export default WorkforcePage;
