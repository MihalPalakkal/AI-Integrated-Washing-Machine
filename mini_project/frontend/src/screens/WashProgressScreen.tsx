import React, { useEffect, useState, useRef } from 'react';
import { View, Text, ScrollView, RefreshControl } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Activity, ShieldCheck } from 'lucide-react-native';
import tw from '../lib/tailwind';
import { api } from '../services/api';
import { onMachineStateChange, onLogsChange, onRootParamsChange } from '../services/firebase';
import { MachineState } from '../types/Machine';
import VerticalStepIndicator from '../components/VerticalStepIndicator';
import { useTheme } from '../context/ThemeContext';

export default function WashProgressScreen() {
  const insets = useSafeAreaInsets();
  const { isDarkMode } = useTheme();
  const [machine, setMachine] = useState<MachineState | null>(null);
  const [logs, setLogs] = useState<Record<string, any>>({});
  const [loading, setLoading] = useState(false);

  // Buffer to strictly hold root AI values so they consistently override the state refresh
  const aiParamsRef = useRef<any>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
    const data = await api.getMachineStatus();
    if (data && aiParamsRef.current) {
      data.waterUsage = aiParamsRef.current.water_level ?? data.waterUsage;
      data.detergentUsage = aiParamsRef.current.detergent_amount ?? data.detergentUsage;
      data.temperature = aiParamsRef.current.temperature ?? data.temperature;
      data.loadWeight = aiParamsRef.current.load_weight ?? data.loadWeight;
    }
    setMachine(data);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData(); // Initial REST load
    
    // Firebase real-time listener for machine state
    const unsubState = onMachineStateChange((data) => {
      if (data) {
        if (aiParamsRef.current) {
          data.waterUsage = aiParamsRef.current.water_level ?? data.waterUsage;
          data.detergentUsage = aiParamsRef.current.detergent_amount ?? data.detergentUsage;
          data.temperature = aiParamsRef.current.temperature ?? data.temperature;
          data.loadWeight = aiParamsRef.current.load_weight ?? data.loadWeight;
        }
        setMachine(data);
      }
    });

    // Firebase real-time listener for hardware logs to drive UI stages
    const unsubLogs = onLogsChange((newLogs) => {
      if (newLogs) setLogs(newLogs);
    });

    const unsubRootParams = onRootParamsChange((params) => {
      if (params) {
        aiParamsRef.current = params;
        setMachine((prev) => {
          if (!prev) return prev;
          return {
            ...prev,
            waterUsage: params.water_level ?? prev.waterUsage,
            detergentUsage: params.detergent_amount ?? prev.detergentUsage,
            temperature: params.temperature ?? prev.temperature,
            loadWeight: params.load_weight ?? prev.loadWeight,
          };
        });
      }
    });

    return () => {
      unsubState();
      unsubLogs();
      unsubRootParams();
    };
  }, []);

  return (
    <ScrollView
      style={tw`flex-1 ${isDarkMode ? 'bg-dark-bg' : 'bg-surface'}`}
      contentContainerStyle={{ paddingTop: insets.top, paddingBottom: 100 }}
      refreshControl={<RefreshControl refreshing={loading} onRefresh={fetchData} />}
    >
      <View style={tw`p-5`}>
        {/* Header */}
        <View style={tw`flex-row items-center justify-between mb-8`}>
          <View style={tw`flex-row items-center gap-3`}>
            <View style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-primary/10'} p-2.5 rounded-xl`}>
              <Activity size={22} color="#2A7FFF" />
            </View>
            <View>
              <Text style={tw`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Live Status</Text>
              <Text style={tw`text-gray-400 text-xs`}>Hardware Stream Active</Text>
            </View>
          </View>
          {machine?.status === 'completed' && (
            <View style={tw`bg-emerald-500/10 px-3 py-1.5 rounded-lg flex-row items-center gap-1.5`}>
              <ShieldCheck size={14} color="#10B981" />
              <Text style={tw`text-emerald-500 text-xs font-bold`}>FINISHED</Text>
            </View>
          )}
        </View>

        {machine ? (
          <>
            {/* HER0 STATS GRID - Now at the Top */}
            <View style={tw`flex-row flex-wrap gap-3 mb-8`}>
              <StatBox label="Water Used" value={`${machine.waterUsage} L`} color="#2A7FFF" isDarkMode={isDarkMode} />
              <StatBox label="Detergent" value={`${machine.detergentUsage} ml`} color="#00C9A7" isDarkMode={isDarkMode} />
              <StatBox label="Temperature" value={`${machine.temperature}°C`} color="#FF6B6B" isDarkMode={isDarkMode} />
              <StatBox label="Load" value={`${machine.loadWeight} kg`} color="#8B5CF6" isDarkMode={isDarkMode} />
            </View>

            {/* OVERALL PROGRESS BAR */}
            <View style={tw`mb-8`}>
              <View style={tw`flex-row justify-between mb-2`}>
                <Text style={tw`text-sm font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Overall Progress</Text>
                <Text style={tw`text-sm font-bold text-primary`}>
                  {machine.status === 'idle' ? '0' : Math.min(100, Math.round((Object.keys(logs).length / 16) * 100))}%
                </Text>
              </View>
              <View style={tw`h-3 w-full rounded-full ${isDarkMode ? 'bg-dark-border' : 'bg-gray-200'} overflow-hidden`}>
                <View 
                  style={[
                    tw`h-full bg-primary rounded-full`, 
                    { width: `${machine.status === 'idle' ? 0 : Math.min(100, Math.round((Object.keys(logs).length / 16) * 100))}%` }
                  ]} 
                />
              </View>
            </View>

            {/* VERTICAL WASH STAGES - Real-time from Logs */}
            <View style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-white'} rounded-3xl p-6 shadow-sm`}>
              <View style={tw`flex-row justify-between items-center mb-6`}>
                <Text style={tw`text-base font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Operational Timeline</Text>
                <Text style={tw`text-[10px] text-gray-400 font-bold uppercase tracking-widest`}>{machine.currentCycle}</Text>
              </View>
              
              <VerticalStepIndicator logs={logs} machineStatus={machine.status} />
            </View>
          </>
        ) : (
          <View style={tw`items-center py-20`}>
            <Text style={tw`text-gray-400`}>Connecting to telemetry...</Text>
          </View>
        )}
      </View>
    </ScrollView>
  );
}

function StatBox({ label, value, color, isDarkMode }: { label: string; value: string; color: string; isDarkMode: boolean }) {
  return (
    <View style={[tw`${isDarkMode ? 'bg-dark-card' : 'bg-white'} rounded-2xl p-4 shadow-sm`, { width: '48%' }]}>
      <View style={[tw`w-8 h-1 rounded-full mb-3`, { backgroundColor: color }]} />
      <Text style={tw`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>{value}</Text>
      <Text style={tw`text-gray-400 text-xs mt-0.5 font-medium`}>{label}</Text>
    </View>
  );
}
