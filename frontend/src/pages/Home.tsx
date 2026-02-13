import { ArrowRight, Activity, Zap, Cpu } from 'lucide-react';
import { Link } from 'react-router-dom';

const Home = () => {
    return (
        <div className="max-w-7xl mx-auto space-y-24 pt-12 pb-24">
            {/* Hero Section */}
            <div className="relative text-center space-y-8">
                <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-sm font-medium animate-fade-in-up">
                    <Zap size={16} />
                    <span>Next-Gen Process Intelligence</span>
                </div>

                <h1 className="text-6xl md:text-8xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-b from-white via-white/90 to-white/50 animate-fade-in-up delay-100">
                    Digital Twin <br />
                    <span className="text-primary drop-shadow-[0_0_30px_rgba(6,182,212,0.3)]">Simulation</span>
                </h1>

                <p className="max-w-2xl mx-auto text-xl text-gray-400 animate-fade-in-up delay-200">
                    Optimize your business processes with AI-driven insights. Simulate scenarios, identify bottlenecks, and automate decision-making in real-time.
                </p>

                <div className="flex justify-center gap-6 pt-4 animate-fade-in-up delay-300">
                    <Link
                        to="/topology"
                        className="group relative px-8 py-4 bg-primary text-background font-bold rounded-xl overflow-hidden hover:scale-105 transition-transform"
                    >
                        <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300" />
                        <span className="relative flex items-center gap-2">
                            Launch Simulation <ArrowRight size={20} />
                        </span>
                    </Link>

                    <Link
                        to="/telemetry"
                        className="px-8 py-4 bg-surface border border-white/10 rounded-xl font-medium hover:bg-white/5 hover:border-white/20 transition-colors"
                    >
                        View Live Telemetry
                    </Link>
                </div>
            </div>

            {/* Feature Grid */}
            <div className="grid md:grid-cols-3 gap-8">
                {[
                    {
                        icon: Activity,
                        title: "Real-time Analytics",
                        desc: "Monitor KPIs with millisecond precision using our advanced telemetry dashboard."
                    },
                    {
                        icon: Cpu,
                        title: "AI Optimization",
                        desc: "Leverage Graph Neural Networks to predict and resolve process bottlenecks automatically."
                    },
                    {
                        icon: Zap,
                        title: "Instant Simulation",
                        desc: "Drag-and-drop resources to instantly simulate outcomes before deployment."
                    }
                ].map((feature, i) => (
                    <div
                        key={i}
                        className="p-8 rounded-2xl bg-surface/30 border border-white/5 backdrop-blur-sm hover:border-primary/30 transition-colors group"
                    >
                        <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center text-primary mb-6 group-hover:scale-110 transition-transform">
                            <feature.icon size={24} />
                        </div>
                        <h3 className="text-xl font-bold mb-3">{feature.title}</h3>
                        <p className="text-gray-400 leading-relaxed">{feature.desc}</p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Home;
