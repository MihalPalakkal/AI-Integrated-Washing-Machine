import React, { useState } from 'react';
import { View, Text, ScrollView, Switch, TouchableOpacity, TextInput, Alert } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Settings, Wifi, Bell, Moon, Info, ChevronRight, Trash2 } from 'lucide-react-native';
import tw from '../lib/tailwind';
import { useTheme } from '../context/ThemeContext';
import { clearAllHistory } from '../services/firebase';

export default function SettingsScreen() {
  const insets = useSafeAreaInsets();
  const { isDarkMode, notificationsEnabled, toggleDarkMode, toggleNotifications } = useTheme();
  const [apiUrl, setApiUrl] = useState('http://localhost:8001');
  const [editing, setEditing] = useState(false);

  const handleSaveUrl = () => {
    setEditing(false);
    Alert.alert('Saved', `API URL changed to: ${apiUrl}`);
  };

  const handleClearHistory = () => {
    Alert.alert(
      'Clear All History',
      'This will delete all wash history, fabric analysis records, and notifications from Firebase. This cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          style: 'destructive',
          onPress: async () => {
            await clearAllHistory();
            Alert.alert('Done', 'All history has been cleared.');
          },
        },
      ]
    );
  };

  return (
    <ScrollView
      style={tw`${isDarkMode ? 'bg-dark-bg' : 'bg-surface'}`}
      contentContainerStyle={{ paddingTop: insets.top, paddingBottom: 100 }}
    >
      <View style={tw`p-5`}>
        {/* Header */}
        <View style={tw`flex-row items-center gap-3 mb-6`}>
          <View style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-gray-200'} p-2.5 rounded-xl`}>
            <Settings size={22} color={isDarkMode ? '#94A3B8' : '#64748B'} />
          </View>
          <View>
            <Text style={tw`text-2xl font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Settings</Text>
            <Text style={tw`text-gray-400 text-xs`}>App configuration</Text>
          </View>
        </View>

        {/* Machine Connection */}
        <Text style={tw`text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 ml-1`}>
          Connection
        </Text>
        <View style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-white'} rounded-2xl p-5 mb-5`}>
          <View style={tw`flex-row items-center gap-3 mb-3`}>
            <Wifi size={18} color="#2A7FFF" />
            <Text style={tw`text-base font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Machine Connection</Text>
          </View>
          {editing ? (
            <View style={tw`gap-3`}>
              <TextInput
                value={apiUrl}
                onChangeText={setApiUrl}
                style={tw`${isDarkMode ? 'bg-slate-700 text-white' : 'bg-gray-50 text-gray-800'} px-4 py-3 rounded-xl border ${isDarkMode ? 'border-dark-border' : 'border-gray-200'}`}
                placeholder="http://192.168.x.x:8001"
                placeholderTextColor={isDarkMode ? '#64748B' : '#94A3B8'}
                autoCapitalize="none"
                autoCorrect={false}
              />
              <View style={tw`flex-row gap-2`}>
                <TouchableOpacity
                  onPress={handleSaveUrl}
                  style={tw`flex-1 bg-primary py-3 rounded-xl items-center`}
                >
                  <Text style={tw`text-white font-bold`}>Save</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => setEditing(false)}
                  style={tw`flex-1 ${isDarkMode ? 'bg-dark-border' : 'bg-gray-100'} py-3 rounded-xl items-center`}
                >
                  <Text style={tw`${isDarkMode ? 'text-dark-text' : 'text-gray-600'} font-bold`}>Cancel</Text>
                </TouchableOpacity>
              </View>
            </View>
          ) : (
            <TouchableOpacity
              onPress={() => setEditing(true)}
              style={tw`flex-row justify-between items-center`}
            >
              <Text style={tw`text-gray-500 text-sm flex-1`}>{apiUrl}</Text>
              <ChevronRight size={16} color="#94A3B8" />
            </TouchableOpacity>
          )}
        </View>

        {/* Preferences */}
        <Text style={tw`text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 ml-1`}>
          Preferences
        </Text>
        <View style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-white'} rounded-2xl mb-5`}>
          <ToggleRow
            icon={<Bell size={18} color="#F59E0B" />}
            label="Notifications"
            description="Receive wash alerts"
            value={notificationsEnabled}
            onToggle={toggleNotifications}
            isDarkMode={isDarkMode}
          />
          <View style={tw`border-t ${isDarkMode ? 'border-dark-border' : 'border-gray-50'}`} />
          <ToggleRow
            icon={<Moon size={18} color="#6366F1" />}
            label="Dark Mode"
            description="Toggle app theme"
            value={isDarkMode}
            onToggle={toggleDarkMode}
            isDarkMode={isDarkMode}
          />
        </View>

        {/* Danger Zone */}
        <Text style={tw`text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 ml-1`}>
          Danger Zone
        </Text>
        <TouchableOpacity
          onPress={handleClearHistory}
          style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-white'} rounded-2xl p-5 mb-5 flex-row items-center gap-3`}
        >
          <Trash2 size={18} color="#FF6B6B" />
          <View style={tw`flex-1`}>
            <Text style={tw`text-base font-bold text-red-500`}>Clear All History</Text>
            <Text style={tw`text-gray-400 text-xs`}>Delete wash history, analysis records & notifications</Text>
          </View>
        </TouchableOpacity>

        {/* About */}
        <Text style={tw`text-xs font-bold text-gray-400 uppercase tracking-wider mb-2 ml-1`}>
          About
        </Text>
        <View style={tw`${isDarkMode ? 'bg-dark-card' : 'bg-white'} rounded-2xl p-5`}>
          <View style={tw`flex-row items-center gap-3 mb-3`}>
            <Info size={18} color="#94A3B8" />
            <Text style={tw`text-base font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>App Information</Text>
          </View>
          <InfoRow label="App Name" value="Smart Washer" isDarkMode={isDarkMode} />
          <InfoRow label="Version" value="1.0.0" isDarkMode={isDarkMode} />
          <InfoRow label="Database" value="Firebase RTDB" isDarkMode={isDarkMode} />
          <InfoRow label="Framework" value="React Native (Expo)" isDarkMode={isDarkMode} />
        </View>
      </View>
    </ScrollView>
  );
}

function ToggleRow({
  icon,
  label,
  description,
  value,
  onToggle,
  isDarkMode,
}: {
  icon: React.ReactNode;
  label: string;
  description: string;
  value: boolean;
  onToggle: (v: boolean) => void;
  isDarkMode: boolean;
}) {
  return (
    <View style={tw`flex-row justify-between items-center p-5`}>
      <View style={tw`flex-row items-center gap-3 flex-1`}>
        {icon}
        <View>
          <Text style={tw`text-base font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>{label}</Text>
          <Text style={tw`text-gray-400 text-xs`}>{description}</Text>
        </View>
      </View>
      <Switch
        value={value}
        onValueChange={onToggle}
        trackColor={{ true: '#2A7FFF', false: isDarkMode ? '#334155' : '#E2E8F0' }}
        thumbColor="white"
      />
    </View>
  );
}

function InfoRow({ label, value, isDarkMode }: { label: string; value: string; isDarkMode: boolean }) {
  return (
    <View style={tw`flex-row justify-between py-2`}>
      <Text style={tw`text-gray-500 text-sm`}>{label}</Text>
      <Text style={tw`${isDarkMode ? 'text-slate-300' : 'text-gray-800'} font-medium text-sm`}>{value}</Text>
    </View>
  );
}
