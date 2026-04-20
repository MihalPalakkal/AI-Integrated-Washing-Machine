import React, { useState, useEffect } from 'react';
import { View, Text, ScrollView, TouchableOpacity, Switch, Alert, ActivityIndicator } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Sliders, Send, RotateCcw, ChevronLeft } from 'lucide-react-native';
import tw from '../lib/tailwind';
import { api } from '../services/api';
import { WashCycle } from '../types/Cycle';
import { useTheme } from '../context/ThemeContext';

const TEMP_OPTIONS = [20, 30, 40, 60, 90];
const DURATION_OPTIONS = [30, 45, 60, 75, 90];
const WATER_OPTIONS = [10, 20, 30, 40, 50, 60];
const DETERGENT_OPTIONS = [20, 25, 30, 35, 40, 50];

export default function ManualOverrideScreen({ navigation }: any) {
  const insets = useSafeAreaInsets();
  const { isDarkMode } = useTheme();
  const lastCycle = api.getLastPredictedCycle();
  
  const [spinOptions, setSpinOptions] = useState<number[]>([5, 10, 15, 20, 25]);
  const [cycle, setCycle] = useState<WashCycle>({
    temperature: 30,
    spinTime: 10,
    duration: 45,
    detergent: 35,
    water: 20,
    soakTime: 10,
    washCycles: 1,
    agitationPattern: 'Normal',
  });
  const [extraRinse, setExtraRinse] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (lastCycle) {
      setCycle({
        ...lastCycle,
        temperature: lastCycle.temperature || 30,
        spinTime: lastCycle.spinTime || 10,
        duration: lastCycle.duration || 45,
        detergent: lastCycle.detergent || 35,
        water: lastCycle.water || 20,
        soakTime: lastCycle.soakTime || 10,
        washCycles: lastCycle.washCycles || 1,
        agitationPattern: lastCycle.agitationPattern || 'Normal',
      });
      if (Array.isArray(lastCycle.spinTimeOptions) && lastCycle.spinTimeOptions.length > 0) {
        setSpinOptions(lastCycle.spinTimeOptions);
      }
    }
  }, [lastCycle]);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      await api.overrideCycle({ ...cycle, extraRinse });
      Alert.alert('Success', 'Override settings applied to the machine!');
      navigation.goBack();
    } catch {
      Alert.alert('Error', 'Failed to apply settings. Is the machine connected?');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    if (lastCycle) {
      setCycle({ ...lastCycle });
    } else {
      setCycle({
        temperature: 30,
        spinTime: 10,
        duration: 45,
        detergent: 35,
        water: 20,
        soakTime: 10,
        washCycles: 1,
        agitationPattern: 'Normal',
      });
    }
  };

  return (
    <View style={tw`flex-1 ${isDarkMode ? 'bg-dark-bg' : 'bg-surface'}`}>
      <ScrollView
        contentContainerStyle={{ paddingTop: insets.top + 20, paddingBottom: 120 }}
        style={tw`flex-1`}
      >
        <View style={tw`px-5`}>
          {/* Header */}
          <View style={tw`flex-row items-center justify-between mb-6`}>
            <View style={tw`flex-row items-center gap-3`}>
              <TouchableOpacity 
                onPress={() => navigation.goBack()}
                style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-white'} p-2 rounded-xl mr-2`}
              >
                <ChevronLeft size={20} color={isDarkMode ? '#94A3B8' : '#64748B'} />
              </TouchableOpacity>
              <View>
                <Text style={tw`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Manual Override</Text>
                <Text style={tw`text-gray-400 text-xs`}>Customize AI predicted settings</Text>
              </View>
            </View>
            <TouchableOpacity onPress={handleReset} style={tw`p-2`}>
              <RotateCcw size={20} color="#2A7FFF" />
            </TouchableOpacity>
          </View>

          {/* Temperature */}
          <SectionCard title="Temperature" value={`${cycle.temperature}°C`} isDarkMode={isDarkMode}>
            <View style={tw`flex-row gap-2 flex-wrap`}>
              {TEMP_OPTIONS.map((t) => (
                <OptionChip
                  key={t}
                  label={`${t}°C`}
                  selected={cycle.temperature === t}
                  color="#FF6B6B"
                  onPress={() => setCycle({ ...cycle, temperature: t })}
                  isDarkMode={isDarkMode}
                />
              ))}
            </View>
          </SectionCard>

          {/* Spin Time */}
          <SectionCard title="Spin Time" value={`${cycle.spinTime} min`} isDarkMode={isDarkMode}>
            <View style={tw`flex-row gap-2 flex-wrap`}>
              {spinOptions.map((s) => (
                <OptionChip
                  key={s}
                  label={`${s}m`}
                  selected={cycle.spinTime === s}
                  color="#6366F1"
                  onPress={() => setCycle({ ...cycle, spinTime: s })}
                  isDarkMode={isDarkMode}
                />
              ))}
            </View>
          </SectionCard>

          {/* Duration */}
          <SectionCard title="Wash Duration" value={`${cycle.duration} min`} isDarkMode={isDarkMode}>
            <View style={tw`flex-row gap-2 flex-wrap`}>
              {DURATION_OPTIONS.map((d) => (
                <OptionChip
                  key={d}
                  label={`${d}m`}
                  selected={cycle.duration === d}
                  color="#2A7FFF"
                  onPress={() => setCycle({ ...cycle, duration: d })}
                  isDarkMode={isDarkMode}
                />
              ))}
            </View>
          </SectionCard>

          {/* Water Level */}
          <SectionCard title="Water Level" value={`${cycle.water} L`} isDarkMode={isDarkMode}>
            <View style={tw`flex-row gap-2 flex-wrap`}>
              {WATER_OPTIONS.map((w) => (
                <OptionChip
                  key={w}
                  label={`${w}L`}
                  selected={cycle.water === w}
                  color="#00C9A7"
                  onPress={() => setCycle({ ...cycle, water: w })}
                  isDarkMode={isDarkMode}
                />
              ))}
            </View>
          </SectionCard>

          {/* Detergent Amount */}
          <SectionCard title="Detergent Amount" value={`${cycle.detergent} ml`} isDarkMode={isDarkMode}>
            <View style={tw`flex-row gap-2 flex-wrap`}>
              {DETERGENT_OPTIONS.map((d) => (
                <OptionChip
                  key={d}
                  label={`${d}ml`}
                  selected={cycle.detergent === d}
                  color="#8B5CF6"
                  onPress={() => setCycle({ ...cycle, detergent: d })}
                  isDarkMode={isDarkMode}
                />
              ))}
            </View>
          </SectionCard>

          {/* Extra Rinse Toggle */}
          <View style={tw`${isDarkMode ? 'bg-dark-card border border-dark-border' : 'bg-white shadow-sm'} rounded-2xl p-5 mb-4 flex-row justify-between items-center`}>
            <View>
              <Text style={tw`text-base font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Extra Rinse</Text>
              <Text style={tw`text-gray-400 text-xs mt-0.5`}>Add an additional rinse cycle</Text>
            </View>
            <Switch
              value={extraRinse}
              onValueChange={setExtraRinse}
              trackColor={{ true: '#2A7FFF', false: isDarkMode ? '#334155' : '#E2E8F0' }}
              thumbColor="white"
            />
          </View>
        </View>
      </ScrollView>

      {/* Floating Submit Button */}
      <View style={[tw`absolute bottom-0 left-0 right-0 p-5`, { paddingBottom: insets.bottom + 10 }]}>
        <TouchableOpacity
          onPress={handleSubmit}
          disabled={loading}
          style={tw`bg-primary py-4 rounded-2xl flex-row items-center justify-center gap-2 shadow-lg shadow-primary/30`}
        >
          {loading ? (
            <ActivityIndicator color="white" />
          ) : (
            <>
              <Send size={18} color="white" />
              <Text style={tw`text-white font-bold text-base`}>Apply These Settings</Text>
            </>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

function SectionCard({ title, value, children, isDarkMode }: { title: string; value: string; children: React.ReactNode; isDarkMode: boolean }) {
  return (
    <View style={tw`${isDarkMode ? 'bg-dark-card border border-dark-border' : 'bg-white shadow-sm'} rounded-2xl p-5 mb-4`}>
      <View style={tw`flex-row justify-between items-center mb-3`}>
        <Text style={tw`text-base font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>{title}</Text>
        <Text style={tw`text-primary font-bold text-sm`}>{value}</Text>
      </View>
      {children}
    </View>
  );
}

function OptionChip({
  label,
  selected,
  color,
  onPress,
  isDarkMode,
}: {
  label: string;
  selected: boolean;
  color: string;
  onPress: () => void;
  isDarkMode: boolean;
}) {
  return (
    <TouchableOpacity
      onPress={onPress}
      style={[
        tw`px-4 py-2.5 rounded-xl`,
        selected
          ? { backgroundColor: color }
          : tw`${isDarkMode ? 'bg-slate-800' : 'bg-gray-100'}`,
      ]}
    >
      <Text
        style={[
          tw`font-bold text-sm`,
          selected ? tw`text-white` : tw`${isDarkMode ? 'text-slate-400' : 'text-gray-500'}`,
        ]}
      >
        {label}
      </Text>
    </TouchableOpacity>
  );
}
