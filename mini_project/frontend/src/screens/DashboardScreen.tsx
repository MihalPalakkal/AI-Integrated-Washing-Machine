import React, { useEffect, useState, useRef } from 'react';
import { View, Text, TouchableOpacity, ScrollView, RefreshControl } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Play, Bell, Sparkles, Activity, Link2 } from 'lucide-react-native';
import tw from '../lib/tailwind';
import { api } from '../services/api';
import { onMachineStateChange, getFabricAnalysisHistory, onRootStatusChange, onRootParamsChange } from '../services/firebase';
import { MachineState } from '../types/Machine';
import { AppNotification } from '../types/Cycle';
import MachineStatusCard from '../components/MachineStatusCard';
import NotificationCard from '../components/NotificationCard';
import { useTheme } from '../context/ThemeContext';

export default function DashboardScreen({ navigation }: any) {
  const insets = useSafeAreaInsets();
  const { isDarkMode, notificationsEnabled } = useTheme();
  const [machine, setMachine] = useState<MachineState | null>(null);
  const [notifications, setNotifications] = useState<AppNotification[]>([]);
  const [loading, setLoading] = useState(false);
  const [isFirebaseRunning, setIsFirebaseRunning] = useState(false);

  // Strict buffer for AI parameter overrides
  const aiParamsRef = useRef<any>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [machineData, notifData] = await Promise.all([
        api.getMachineStatus(),
        api.getNotifications(),
      ]);
      if (machineData) {
        if (aiParamsRef.current) {
          machineData.waterUsage = aiParamsRef.current.water_level ?? machineData.waterUsage;
          machineData.detergentUsage = aiParamsRef.current.detergent_amount ?? machineData.detergentUsage;
          machineData.temperature = aiParamsRef.current.temperature ?? machineData.temperature;
          machineData.loadWeight = aiParamsRef.current.load_weight ?? machineData.loadWeight;
        }
        setMachine(machineData);
      }
      if (notifData) setNotifications(notifData.slice(0, 3));
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
    // Firebase listener for root status flag
    const unsubStatus = onRootStatusChange((val) => {
      setIsFirebaseRunning(val);
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
      unsubStatus();
      unsubRootParams();
    };
  }, []);

  const handleStart = () => {
    setLoading(true);
    // Instant fetch from Real-time AI buffer instead of waiting for heavy history query
    const latestParams = aiParamsRef.current || api.getLastPredictedCycle() || undefined;
    
    // Non-blocking fire-and-forget for instant UI response
    api.startWash(latestParams).catch(console.error).finally(() => setLoading(false));
  };

  const handleStop = () => {
    setLoading(true);
    // Non-blocking fire-and-forget for instant UI response
    api.stopWash().catch(console.error).finally(() => setLoading(false));
  };

  if (!machine) {
    return (
      <View style={tw`flex-1 justify-center items-center ${isDarkMode ? 'bg-dark-bg' : 'bg-surface'}`}>
        <Text style={tw`text-gray-400`}>Connecting to machine...</Text>
      </View>
    );
  }

  return (
    <ScrollView
      style={tw`flex-1 ${isDarkMode ? 'bg-dark-bg' : 'bg-surface'}`}
      contentContainerStyle={{ paddingTop: insets.top, paddingBottom: 100 }}
      refreshControl={<RefreshControl refreshing={loading} onRefresh={fetchData} />}
    >
      <View style={tw`p-5`}>
        {/* Header */}
        <View style={tw`flex-row justify-between items-center mb-5`}>
          <View>
            <Text style={tw`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Smart Washer</Text>
            <View style={tw`flex-row items-center gap-1.5 mt-0.5`}>
              <View style={tw`w-2 h-2 rounded-full bg-green-500`} />
              <Text style={tw`text-gray-400 text-xs`}>Firebase Synced</Text>
            </View>
          </View>
          <TouchableOpacity
            onPress={() => navigation.navigate('Settings')}
            style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-white'} p-3 rounded-xl`}
          >
            <Bell size={20} color={isDarkMode ? '#94A3B8' : '#64748B'} />
          </TouchableOpacity>
        </View>

        {/* Status Card */}
        <MachineStatusCard machine={machine} />

        {/* Action Buttons & Status Indicators */}
        <View style={tw`mt-5 bg-${isDarkMode ? 'dark-card' : 'white'} rounded-2xl p-4 shadow-sm`}>
          {/* Connection Indicator */}
          <View style={tw`bg-emerald-500/10 py-3 rounded-xl flex-row items-center justify-center gap-2 mb-3`}>
            <Link2 size={18} color="#10B981" />
            <Text style={tw`text-emerald-600 font-bold text-sm`}>Washing Machine Connected</Text>
          </View>
          
          <View style={tw`flex-row gap-3`}>
            {/* Manual Start/Stop Control */}
            {isFirebaseRunning ? (
              <TouchableOpacity
                onPress={handleStop}
                disabled={loading}
                style={tw`flex-1 bg-alert py-4 rounded-xl flex-row items-center justify-center gap-2`}
              >
                <Activity size={20} color="white" />
                <Text style={tw`text-white font-bold text-base`}>Stop Wash</Text>
              </TouchableOpacity>
            ) : (
              <TouchableOpacity
                onPress={handleStart}
                disabled={loading}
                style={tw`flex-1 bg-primary py-4 rounded-xl flex-row items-center justify-center gap-2`}
              >
                <Play size={20} color="white" fill="white" />
                <Text style={tw`text-white font-bold text-base`}>Start Wash</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>

        {/* Quick Nav to Fabric Analysis */}
        <TouchableOpacity
          onPress={() => navigation.navigate('FabricAnalysis')}
          style={tw`mt-5 ${isDarkMode ? 'bg-dark-card' : 'bg-white'} p-4 rounded-2xl flex-row items-center gap-4`}
        >
          <View style={tw`bg-primary/10 p-3 rounded-xl`}>
            <Sparkles size={22} color="#2A7FFF" />
          </View>
          <View style={tw`flex-1`}>
            <Text style={tw`font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>AI Fabric Analysis</Text>
            <Text style={tw`text-gray-400 text-xs mt-0.5`}>
              View detected fabrics & recommended cycle
            </Text>
          </View>
        </TouchableOpacity>

        {/* Recent Notifications */}
        {notificationsEnabled && notifications.length > 0 && (
          <View style={tw`mt-6`}>
            <Text style={tw`text-lg font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'} mb-3`}>Recent Alerts</Text>
            {notifications.map((n) => (
              <NotificationCard key={n.id} notification={n} />
            ))}
          </View>
        )}
      </View>
    </ScrollView>
  );
}
