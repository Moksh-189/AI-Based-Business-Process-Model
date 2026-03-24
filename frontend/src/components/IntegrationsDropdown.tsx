import React, { useState, useEffect, useRef } from 'react';
import { Cloud, LayoutDashboard, MessageSquare, Database, Briefcase, Plus, Check, Loader2, X } from 'lucide-react';

const INTEGRATIONS = [
    {
        id: 'salesforce',
        name: 'Salesforce',
        icon: Cloud,
        fields: [
            { name: 'instanceUrl', label: 'Instance URL' },
            { name: 'clientId', label: 'Client ID' },
            { name: 'clientSecret', label: 'Client Secret', type: 'password' }
        ]
    },
    {
        id: 'jira',
        name: 'Jira Software',
        icon: LayoutDashboard,
        fields: [
            { name: 'domain', label: 'Jira Domain' },
            { name: 'email', label: 'Account Email' },
            { name: 'apiToken', label: 'API Token', type: 'password' }
        ]
    },
    {
        id: 'teams',
        name: 'Microsoft Teams',
        icon: MessageSquare,
        fields: [
            { name: 'tenantId', label: 'Tenant ID' },
            { name: 'clientId', label: 'Client ID' },
            { name: 'clientSecret', label: 'Client Secret', type: 'password' }
        ]
    },
    {
        id: 'sap',
        name: 'SAP ERP',
        icon: Database,
        fields: [
            { name: 'odataUrl', label: 'OData URL' },
            { name: 'username', label: 'Username' },
            { name: 'password', label: 'Password', type: 'password' }
        ]
    },
    {
        id: 'servicenow',
        name: 'ServiceNow',
        icon: Briefcase,
        fields: [
            { name: 'instance', label: 'Instance Name' },
            { name: 'username', label: 'API User' },
            { name: 'password', label: 'API Password', type: 'password' }
        ]
    }
];

const IntegrationsDropdown = () => {
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const [activeModal, setActiveModal] = useState<typeof INTEGRATIONS[0] | null>(null);
    const [formData, setFormData] = useState<Record<string, string>>({});
    const [isConnecting, setIsConnecting] = useState(false);
    const [connectedApps, setConnectedApps] = useState<string[]>([]);

    const dropdownRef = useRef<HTMLDivElement>(null);

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
                setIsDropdownOpen(false);
            }
        };

        if (isDropdownOpen) {
            document.addEventListener('mousedown', handleClickOutside);
        }
        return () => {
            document.removeEventListener('mousedown', handleClickOutside);
        };
    }, [isDropdownOpen]);

    const handleConnect = (e: React.FormEvent) => {
        e.preventDefault();
        if (!activeModal) return;

        setIsConnecting(true);

        // Simulate API connection delay
        setTimeout(() => {
            setConnectedApps(prev => [...prev, activeModal.id]);
            setIsConnecting(false);
            setActiveModal(null);
            setFormData({});
            setIsDropdownOpen(false);
        }, 1500);
    };

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    return (
        <div className="relative" ref={dropdownRef}>
            {/* Trigger Button */}
            <button
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors border border-slate-700 shadow-sm"
            >
                <Plus className="w-4 h-4 text-cyan-400" />
                <span className="font-medium text-sm">Connect +</span>
            </button>

            {/* Dropdown Menu */}
            {isDropdownOpen && (
                <div className="absolute top-full left-0 mt-2 w-64 bg-slate-900 border border-slate-800 rounded-lg shadow-xl overflow-hidden z-50">
                    <div className="p-2">
                        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-3 py-2">
                            Available Integrations
                        </h3>
                        <div className="space-y-1">
                            {INTEGRATIONS.map((app) => {
                                const Icon = app.icon;
                                const isConnected = connectedApps.includes(app.id);

                                return (
                                    <button
                                        key={app.id}
                                        onClick={() => {
                                            if (!isConnected) {
                                                setActiveModal(app);
                                                setFormData({});
                                            }
                                        }}
                                        disabled={isConnected}
                                        className={`w-full flex items-center justify-between px-3 py-2.5 rounded-md transition-colors ${isConnected
                                                ? 'opacity-60 cursor-default'
                                                : 'hover:bg-slate-800 cursor-pointer'
                                            }`}
                                    >
                                        <div className="flex items-center gap-3">
                                            <Icon className={`w-4 h-4 ${isConnected ? 'text-slate-500' : 'text-cyan-400'}`} />
                                            <span className="text-sm font-medium text-slate-200">{app.name}</span>
                                        </div>
                                        {isConnected && <Check className="w-4 h-4 text-emerald-400" />}
                                    </button>
                                );
                            })}
                        </div>
                    </div>
                </div>
            )}

            {/* Connection Modal */}
            {activeModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm">
                    <div className="bg-[#0F172A] border border-slate-800 rounded-xl shadow-2xl w-full max-w-md overflow-hidden animate-in fade-in zoom-in-95 duration-200">
                        {/* Modal Header */}
                        <div className="flex items-center justify-between p-6 border-b border-slate-800 bg-slate-900/50">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-slate-800 rounded-lg">
                                    <activeModal.icon className="w-5 h-5 text-cyan-400" />
                                </div>
                                <h2 className="text-lg font-semibold text-white">Connect {activeModal.name}</h2>
                            </div>
                            <button
                                onClick={() => !isConnecting && setActiveModal(null)}
                                className="text-slate-400 hover:text-white transition-colors p-1"
                                disabled={isConnecting}
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Modal Body */}
                        <form onSubmit={handleConnect} className="p-6 space-y-5">
                            {activeModal.fields.map((field) => (
                                <div key={field.name} className="space-y-1.5">
                                    <label htmlFor={field.name} className="block text-sm font-medium text-slate-300">
                                        {field.label}
                                    </label>
                                    <input
                                        id={field.name}
                                        name={field.name}
                                        type={field.type || 'text'}
                                        required
                                        value={formData[field.name] || ''}
                                        onChange={handleInputChange}
                                        disabled={isConnecting}
                                        className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500 transition-colors disabled:opacity-50"
                                        placeholder={`Enter ${field.label.toLowerCase()}`}
                                    />
                                </div>
                            ))}

                            {/* Modal Footer */}
                            <div className="pt-4 flex justify-end gap-3">
                                <button
                                    type="button"
                                    onClick={() => setActiveModal(null)}
                                    disabled={isConnecting}
                                    className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-white hover:bg-slate-800 rounded-lg transition-colors disabled:opacity-50"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    disabled={isConnecting}
                                    className="flex items-center gap-2 px-6 py-2 bg-[#06B6D4] hover:bg-[#0891B2] text-white text-sm font-medium rounded-lg transition-colors shadow-lg shadow-cyan-500/20 disabled:opacity-70 disabled:cursor-not-allowed"
                                >
                                    {isConnecting ? (
                                        <>
                                            <Loader2 className="w-4 h-4 animate-spin" />
                                            Connecting...
                                        </>
                                    ) : (
                                        'Connect'
                                    )}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
};

export default IntegrationsDropdown;
