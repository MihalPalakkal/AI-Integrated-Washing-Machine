import React from 'react';
import { View, Text } from 'react-native';
import { MachineState } from '../types/Machine';
import { Activity, Droplets, FlaskConical, Thermometer, Weight } from 'lucide-react-native';
import tw from '../lib/tailwind';
import { useTheme } from '../context/ThemeContext';

interface Props {
  machine: MachineState;
}

const statusColors: Record<string, string> = {
  idle: '#00C9A7',
  analyzing: '#2A7FFF',
  filling: '#2A7FFF',
  washing: '#2A7FFF',
  rinsing: '#6366F1',
  spinning: '#8B5CF6',
  completed: '#00C9A7',
  error: '#FF6B6B',
};

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  if (mins === 0 && secs === 0) return '--:--';
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

export default function MachineStatusCard({ machine }: Props) {
  const { isDarkMode } = useTheme();
  const [localTime, setLocalTime] = React.useState(machine.timeRemaining);

  // Sync with machine prop when it updates from backend polling
  React.useEffect(() => {
    setLocalTime(machine.timeRemaining);
  }, [machine.timeRemaining]);

  // Local 1-second countdown for smooth UI
  React.useEffect(() => {
    let interval: NodeJS.Timeout;
    if (machine.status !== 'idle' && machine.status !== 'completed' && localTime > 0) {
      interval = setInterval(() => {
        setLocalTime((prev) => Math.max(0, prev - 1));
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [machine.status, localTime === 0]);

  const color = statusColors[machine.status] || '#94A3B8';
  const isActive = machine.status !== 'idle' && machine.status !== 'completed';

  return (
    <View style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-white'} rounded-3xl p-6 shadow-sm`}>
      {/* Status Header */}
      <View style={tw`items-center mb-5`}>
        <View
          style={[
            tw`px-5 py-2.5 rounded-full mb-3`,
            { backgroundColor: color + '18' },
          ]}
        >
          <Text style={[tw`font-bold text-sm uppercase tracking-wider`, { color }]}>
            {machine.status.replace('_', ' ')}
          </Text>
        </View>

        {isActive && (
          <>
            <Text style={[tw`text-5xl font-bold`, { color }]}>
              {formatTime(localTime)}
            </Text>
            <Text style={tw`text-gray-400 mt-1`}>Time Remaining</Text>
            {machine.currentCycle !== 'None' && (
              <View style={tw`${isDarkMode ? 'bg-dark-border' : 'bg-gray-100'} mt-2 px-4 py-1.5 rounded-full`}>
                <Text style={tw`${isDarkMode ? 'text-dark-text' : 'text-gray-600'} font-medium text-xs`}>
                  {machine.currentCycle}
                </Text>
              </View>
            )}
          </>
        )}

        {!isActive && machine.status === 'idle' && (
          <Text style={tw`text-gray-400 text-base mt-1`}>Ready to wash</Text>
        )}

        {machine.status === 'completed' && (
          <Text style={[tw`text-base font-semibold mt-1`, { color }]}>
            Wash Complete ✓
          </Text>
        )}
      </View>

      {/* Quick Stats Row */}
      <View style={tw`flex-row justify-between`}>
        <StatItem icon={<Droplets size={16} color="#2A7FFF" />} label="Water" value={`${machine.waterUsage}L`} isDarkMode={isDarkMode} />
        <StatItem icon={<FlaskConical size={16} color="#00C9A7" />} label="Detergent" value={`${machine.detergentUsage}ml`} isDarkMode={isDarkMode} />
        <StatItem icon={<Thermometer size={16} color="#FF6B6B" />} label="Temp" value={`${machine.temperature}°C`} isDarkMode={isDarkMode} />
        <StatItem icon={<Weight size={16} color="#8B5CF6" />} label="Load" value={`${machine.loadWeight}kg`} isDarkMode={isDarkMode} />
      </View>
    </View>
  );
}

function StatItem({ icon, label, value, isDarkMode }: { icon: React.ReactNode; label: string; value: string; isDarkMode: boolean }) {
  return (
    <View style={tw`items-center`}>
      <View style={tw`${isDarkMode ? 'bg-dark-border' : 'bg-gray-50'} p-2.5 rounded-xl mb-1.5`}>{icon}</View>
      <Text style={tw`${isDarkMode ? 'text-white' : 'text-gray-800'} font-bold text-sm`}>{value}</Text>
      <Text style={tw`text-gray-400 text-xs`}>{label}</Text>
    </View>
  );
}
