import { useCallback, useEffect } from 'react';
import {
    ReactFlow,
    Background,
    Controls,
    MiniMap,
    useNodesState,
    useEdgesState,
    addEdge,
    type Connection,
    type Edge,
    type Node
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';


const ProcessTopology = () => {
    const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('http://127.0.0.1:8000/api/topology');
                if (!response.ok) {
                    throw new Error('Failed to fetch topology');
                }
                const data = await response.json();
                setNodes(data.nodes || []);
                setEdges(data.edges || []);
            } catch (error) {
                console.error("Error fetching topology:", error);
            }
        };
        fetchData();
    }, [setNodes, setEdges]);

    const onConnect = useCallback((params: Edge | Connection) => setEdges((eds) => addEdge(params, eds)), [setEdges]);

    return (
        <div className="h-[600px] w-full rounded-2xl overflow-hidden border border-white/10 bg-surface/30 backdrop-blur-sm">
            <div className="p-4 border-b border-white/10 flex items-center justify-between bg-primary/5">
                <h3 className="font-bold text-lg text-primary flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                    Live Process Topology (SAP P2P)
                </h3>
                <span className="text-xs text-red-400 border border-red-500/30 bg-red-500/10 px-2 py-1 rounded">
                    Critical Bottleneck Detected
                </span>
            </div>
            <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                fitView
                className="bg-background/50"
            >
                <Controls className="bg-surface text-white border-white/10 fill-white" />
                <MiniMap
                    nodeStrokeColor={(n: Node) => {
                        if (n.style?.background) return n.style.background as string;
                        if (n.type === 'input') return '#0041d0';
                        if (n.type === 'output') return '#ff0072';
                        return '#eee';
                    }}
                    nodeColor={(n: Node) => {
                        if (n.style?.background) return n.style.background as string;
                        return '#fff';
                    }}
                    className="bg-surface border-white/10"
                    maskColor="rgba(0, 0, 0, 0.7)"
                />
                <Background color="#333" gap={16} />
            </ReactFlow>
        </div>
    );
};

export default ProcessTopology;
