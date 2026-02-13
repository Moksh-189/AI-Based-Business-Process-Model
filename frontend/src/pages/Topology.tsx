import ProcessTopology from '../components/ProcessTopology';
import WorkforceAllocation from '../components/WorkforceAllocation';
import TelemetryPanel from '../components/TelemetryPanel';

const Topology = () => {
    return (
        <div className="flex flex-col gap-6 h-full">
            {/* Top Section: Process Graph */}
            <div className="flex-1 min-h-[500px]">
                <ProcessTopology />
            </div>

            {/* Bottom Section: Controls & Metrics */}
            <div className="flex flex-col md:flex-row gap-6 h-[400px]">
                {/* Left: Workforce Drag & Drop */}
                <div className="flex-1">
                    <WorkforceAllocation />
                </div>

                {/* Right: Telemetry Charts */}
                <div className="flex-1">
                    <TelemetryPanel />
                </div>
            </div>
        </div>
    );
};

export default Topology;
