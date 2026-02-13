import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Network, Users, BarChart2, Menu } from 'lucide-react';

const Sidebar = () => {
    const [isCollapsed, setIsCollapsed] = React.useState(false);

    const navItems = [
        { icon: LayoutDashboard, label: 'Dashboard', path: '/' },
        { icon: Network, label: 'Process Topology', path: '/topology' },
        { icon: Users, label: 'Workforce', path: '/workforce' },
        { icon: BarChart2, label: 'Telemetry', path: '/telemetry' },
    ];

    return (
        <div
            className={`relative h-screen bg-background/80 backdrop-blur-xl border-r border-white/10 transition-all duration-300 ${isCollapsed ? 'w-20' : 'w-64'
                }`}
        >
            <div className="flex items-center justify-between p-4 border-b border-white/10">
                <div className={`font-bold text-xl text-primary tracking-wider ${isCollapsed ? 'hidden' : 'block'}`}>
                    AI.BPI
                </div>
                <button
                    onClick={() => setIsCollapsed(!isCollapsed)}
                    className="p-2 hover:bg-white/5 rounded-lg text-gray-400 hover:text-white transition-colors"
                >
                    <Menu size={20} />
                </button>
            </div>

            <nav className="p-4 space-y-2">
                {navItems.map((item) => (
                    <NavLink
                        key={item.path}
                        to={item.path}
                        className={({ isActive }) => `
              flex items-center gap-3 p-3 rounded-xl transition-all duration-200 group
              ${isActive
                                ? 'bg-primary/20 text-primary shadow-[0_0_15px_rgba(6,182,212,0.3)] border border-primary/30'
                                : 'text-gray-400 hover:bg-white/5 hover:text-white'
                            }
            `}
                    >
                        <item.icon size={20} className="min-w-[20px]" />
                        <span className={`${isCollapsed ? 'hidden' : 'block'} font-medium`}>
                            {item.label}
                        </span>

                        {/* Hover Tooltip for Collapsed State */}
                        {isCollapsed && (
                            <div className="absolute left-full ml-4 px-3 py-1 bg-surface border border-white/10 rounded-md text-sm text-white opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50">
                                {item.label}
                            </div>
                        )}
                    </NavLink>
                ))}
            </nav>

            {/* Connection Status Indicator */}
            <div className="absolute bottom-4 left-4 right-4">
                <div className={`flex items-center gap-3 p-3 rounded-xl bg-surface/50 border border-white/5 ${isCollapsed ? 'justify-center' : ''}`}>
                    <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_10px_#10b981] animate-pulse"></div>
                    <span className={`text-xs text-gray-400 ${isCollapsed ? 'hidden' : 'block'}`}>
                        System Online
                    </span>
                </div>
            </div>
        </div>
    );
};

export default Sidebar;
