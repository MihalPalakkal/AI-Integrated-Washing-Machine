import React from 'react';
import { View, Text } from 'react-native';
import { AppNotification } from '../types/Cycle';
import { CheckCircle, AlertTriangle, Info, XCircle } from 'lucide-react-native';
import tw from '../lib/tailwind';
import { useTheme } from '../context/ThemeContext';

interface Props {
  notification: AppNotification;
}

const typeConfig: Record<string, { color: string; bgColor: string; Icon: any }> = {
  success: { color: '#00C9A7', bgColor: '#00C9A715', Icon: CheckCircle },
  warning: { color: '#F59E0B', bgColor: '#F59E0B15', Icon: AlertTriangle },
  info: { color: '#2A7FFF', bgColor: '#2A7FFF15', Icon: Info },
  error: { color: '#FF6B6B', bgColor: '#FF6B6B15', Icon: XCircle },
};

export default function NotificationCard({ notification }: Props) {
  const { isDarkMode } = useTheme();
  const config = typeConfig[notification.type] || typeConfig.info;
  const { Icon } = config;

  return (
    <View style={[
      tw`${isDarkMode ? 'bg-dark-card' : 'bg-white'} rounded-2xl p-4 mb-3 flex-row items-start gap-3`, 
      { shadowColor: '#000', shadowOpacity: 0.03, shadowRadius: 8, elevation: 1 }
    ]}>
      <View style={[tw`p-2.5 rounded-xl`, { backgroundColor: isDarkMode ? config.color + '25' : config.bgColor }]}>
        <Icon size={20} color={config.color} />
      </View>
      <View style={tw`flex-1`}>
        <Text style={tw`font-bold ${isDarkMode ? 'text-white' : 'text-gray-800'} text-sm`}>{notification.title}</Text>
        <Text style={tw`${isDarkMode ? 'text-dark-muted' : 'text-gray-500'} text-xs mt-0.5 leading-4`}>{notification.message}</Text>
        <Text style={tw`text-gray-400 text-[10px] mt-2`}>
          {new Date(notification.timestamp).toLocaleString()}
        </Text>
      </View>
    </View>
  );
}
