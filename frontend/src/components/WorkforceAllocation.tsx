import { useState, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { API_URL } from '../config/api';
import { useToast } from '../context/ToastContext';
import { DragDropContext, Droppable, Draggable, type DropResult, type DraggableProvided, type DraggableStateSnapshot } from '@hello-pangea/dnd';
import { Users, User, AlertCircle } from 'lucide-react';

// Portal wrapper to fix drag offset caused by CSS transforms (backdrop-blur, etc.)
const PortalAwareDraggable = ({ provided, snapshot, children }: {
    provided: DraggableProvided;
    snapshot: DraggableStateSnapshot;
    children: React.ReactNode;
}) => {
    const child = (
        <div
            ref={provided.innerRef}
            {...provided.draggableProps}
            {...provided.dragHandleProps}
        >
            {children}
        </div>
    );

    if (snapshot.isDragging) {
        return createPortal(child, document.body);
    }
    return child;
};
interface Employee {
    id: string;
    name: string;
    role: string;
    efficiency: number; // 0-100
}

const initialEmployees = [
    { id: 'emp-1', name: 'Alice Chen', role: 'Snr. Analyst', efficiency: 95 },
    { id: 'emp-2', name: 'Bob Smith', role: 'Approver', efficiency: 78 },
    { id: 'emp-3', name: 'Charlie D', role: 'Clerk', efficiency: 82 },
    { id: 'emp-4', name: 'Dana Lee', role: 'QA Engineer', efficiency: 88 },
    { id: 'emp-5', name: 'Eve Park', role: 'Snr. Approver', efficiency: 91 },
    { id: 'emp-6', name: 'Frank Wu', role: 'Analyst', efficiency: 75 },
];

const WorkforceAllocation = () => {
    const [availableStaff, setAvailableStaff] = useState<Employee[]>(initialEmployees);
    const [assignedStaff, setAssignedStaff] = useState<Employee[]>([]);

    const onDragEnd = (result: DropResult) => {
        if (!result.destination) return;

        const sourceList = result.source.droppableId === 'available' ? availableStaff : assignedStaff;
        const destList = result.destination.droppableId === 'available' ? availableStaff : assignedStaff;

        // Function to reorder
        if (result.source.droppableId === result.destination.droppableId) {
            const items = Array.from(sourceList);
            const [reorderedItem] = items.splice(result.source.index, 1);
            items.splice(result.destination.index, 0, reorderedItem);

            if (result.source.droppableId === 'available') setAvailableStaff(items);
            else setAssignedStaff(items);
        }
        // Function to move between lists
        else {
            const sourceClone = Array.from(sourceList);
            const destClone = Array.from(destList);
            const [movedItem] = sourceClone.splice(result.source.index, 1);
            destClone.splice(result.destination.index, 0, movedItem);

            if (result.source.droppableId === 'available') {
                setAvailableStaff(sourceClone);
                setAssignedStaff(destClone);
            } else {
                setAssignedStaff(sourceClone);
                setAvailableStaff(destClone);
            }
        }
    };

    const { showToast } = useToast();

    // Trigger simulation when assignment changes
    useEffect(() => {
        const updateAllocation = async () => {
            try {
                // Digital Twin Simulation call
                await fetch(`${API_URL}/api/simulate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ assigned: assignedStaff })
                });
                showToast("Digital Twin simulation updated.", "success", 2000);
            } catch (error) {
                console.error("Simulation update failed:", error);
                showToast("Simulation sync failed.", "error");
            }
        };

        if (assignedStaff.length > 0) {
            updateAllocation();
        }
    }, [assignedStaff]);

    return (
        <div className="h-[600px] flex flex-col bg-surface/30 backdrop-blur-sm border border-white/10 rounded-2xl overflow-hidden">
            <div className="p-4 border-b border-white/10 bg-primary/5 flex justify-between items-center">
                <h3 className="font-bold text-lg text-primary flex items-center gap-2">
                    <Users size={20} /> Workforce Allocation
                </h3>
                <span className="text-xs text-gray-400">Drag to assign</span>
            </div>

            <DragDropContext onDragEnd={onDragEnd}>
                <div className="flex-1 overflow-auto p-4 space-y-6">

                    {/* Assigned Zone (Target for Optimization) */}
                    <div className="space-y-3">
                        <h4 className="text-sm font-medium text-red-400 flex items-center gap-2">
                            <AlertCircle size={14} /> BottleNeck Node: Clear Invoice
                        </h4>
                        <Droppable droppableId="assigned">
                            {(provided, snapshot) => (
                                <div
                                    ref={provided.innerRef}
                                    {...provided.droppableProps}
                                    className={`min-h-[120px] p-3 rounded-xl border-2 border-dashed transition-colors ${snapshot.isDraggingOver
                                        ? 'bg-red-500/10 border-red-500/50'
                                        : 'bg-surface/50 border-white/10'
                                        }`}
                                >
                                    {assignedStaff.length === 0 && !snapshot.isDraggingOver && (
                                        <div className="h-full flex items-center justify-center text-gray-500 text-sm italic">
                                            Drop staff here to optimize
                                        </div>
                                    )}
                                    {assignedStaff.map((emp, index) => (
                                        <Draggable key={emp.id} draggableId={emp.id} index={index}>
                                            {(provided, snapshot) => (
                                                <PortalAwareDraggable provided={provided} snapshot={snapshot}>
                                                    <div
                                                        data-testid="draggable-card"
                                                        data-employee-name={emp.name}
                                                        className="mb-2 p-3 bg-surface border border-white/10 rounded-lg shadow-sm flex justify-between items-center group hover:border-primary/50"
                                                    >
                                                        <div className="flex items-center gap-3">
                                                            <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center text-primary">
                                                                <User size={16} />
                                                            </div>
                                                            <div>
                                                                <div className="font-bold text-sm">{emp.name}</div>
                                                                <div className="text-xs text-gray-400">{emp.role}</div>
                                                            </div>
                                                        </div>
                                                        <div className="text-emerald-400 text-xs font-mono">
                                                            {emp.efficiency}% Eff.
                                                        </div>
                                                    </div>
                                                </PortalAwareDraggable>
                                            )}
                                        </Draggable>
                                    ))}
                                    {provided.placeholder}
                                </div>
                            )}
                        </Droppable>
                    </div>

                    {/* Available Pool */}
                    <div className="space-y-3">
                        <h4 className="text-sm font-medium text-gray-400">Available Resources</h4>
                        <Droppable droppableId="available">
                            {(provided, snapshot) => (
                                <div
                                    ref={provided.innerRef}
                                    {...provided.droppableProps}
                                    className={`min-h-[100px] p-3 rounded-xl transition-colors ${snapshot.isDraggingOver ? 'bg-white/5' : ''
                                        }`}
                                >
                                    {availableStaff.map((emp, index) => (
                                        <Draggable key={emp.id} draggableId={emp.id} index={index}>
                                            {(provided, snapshot) => (
                                                <PortalAwareDraggable provided={provided} snapshot={snapshot}>
                                                    <div
                                                        data-testid="draggable-card"
                                                        data-employee-name={emp.name}
                                                        className="mb-2 p-3 bg-surface/50 border border-white/5 rounded-lg flex justify-between items-center cursor-grab active:cursor-grabbing hover:bg-surface/80"
                                                    >
                                                        <div className="flex items-center gap-3">
                                                            <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-gray-300">
                                                                <User size={16} />
                                                            </div>
                                                            <div>
                                                                <div className="font-bold text-sm">{emp.name}</div>
                                                                <div className="text-xs text-gray-500">{emp.role}</div>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </PortalAwareDraggable>
                                            )}
                                        </Draggable>
                                    ))}
                                    {provided.placeholder}
                                </div>
                            )}
                        </Droppable>
                    </div>

                </div>
            </DragDropContext>
        </div>
    );
};

export default WorkforceAllocation;
