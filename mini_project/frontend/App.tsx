import React, { useEffect } from 'react';
import { NavigationContainer, DefaultTheme, DarkTheme } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { Home, Eye, Loader, Clock, Settings } from 'lucide-react-native';

import DashboardScreen from './src/screens/DashboardScreen';
import FabricAnalysisScreen from './src/screens/FabricAnalysisScreen';
import WashProgressScreen from './src/screens/WashProgressScreen';
import HistoryScreen from './src/screens/HistoryScreen';
import SettingsScreen from './src/screens/SettingsScreen';
import { ThemeProvider, useTheme } from './src/context/ThemeContext';
import { discoverBackendIp } from './src/services/api';

import { createNativeStackNavigator } from '@react-navigation/native-stack';
import ManualOverrideScreen from './src/screens/ManualOverrideScreen';

const Stack = createNativeStackNavigator();
const Tab = createBottomTabNavigator();

function TabNavigator() {
  const { isDarkMode } = useTheme();
  return (
    <Tab.Navigator
        screenOptions={{
          headerShown: false,
          tabBarStyle: {
            backgroundColor: isDarkMode ? '#1E293B' : 'white', // slate-800 vs white
            borderTopWidth: 0,
            elevation: 8,
            shadowOpacity: 0.08,
            shadowRadius: 12,
            height: 65,
            paddingBottom: 10,
            paddingTop: 6,
          },
          tabBarActiveTintColor: '#2A7FFF',
          tabBarInactiveTintColor: isDarkMode ? '#64748B' : '#94A3B8',
          tabBarLabelStyle: {
            fontSize: 11,
            fontWeight: '600',
          },
        }}
      >
        <Tab.Screen
          name="Dashboard"
          component={DashboardScreen}
          options={{
            tabBarIcon: ({ color, size }) => <Home color={color} size={size} />,
            tabBarLabel: 'Dashboard',
          }}
        />
        <Tab.Screen
          name="FabricAnalysis"
          component={FabricAnalysisScreen}
          options={{
            tabBarIcon: ({ color, size }) => <Eye color={color} size={size} />,
            tabBarLabel: 'Fabrics',
          }}
        />
        <Tab.Screen
          name="WashProgress"
          component={WashProgressScreen}
          options={{
            tabBarIcon: ({ color, size }) => <Loader color={color} size={size} />,
            tabBarLabel: 'Progress',
          }}
        />
        <Tab.Screen
          name="History"
          component={HistoryScreen}
          options={{
            tabBarIcon: ({ color, size }) => <Clock color={color} size={size} />,
            tabBarLabel: 'History',
          }}
        />
        <Tab.Screen
          name="Settings"
          component={SettingsScreen}
          options={{
            tabBarIcon: ({ color, size }) => <Settings color={color} size={size} />,
            tabBarLabel: 'Settings',
          }}
        />
      </Tab.Navigator>
  );
}

function AppContent() {
  const { isDarkMode } = useTheme();

  useEffect(() => {
    discoverBackendIp();
  }, []);

  const theme = isDarkMode ? DarkTheme : DefaultTheme;
  const customTheme = {
    ...theme,
    colors: {
      ...theme.colors,
      background: isDarkMode ? '#0F172A' : '#F7F9FC', // slate-900 vs surface
    },
  };

  return (
    <NavigationContainer theme={customTheme}>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="MainTabs" component={TabNavigator} />
        <Stack.Screen 
          name="ManualOverride" 
          component={ManualOverrideScreen} 
          options={{ 
            headerShown: true,
            headerTitle: 'Customize Wash',
            headerStyle: { backgroundColor: isDarkMode ? '#1E293B' : 'white' },
            headerTintColor: isDarkMode ? 'white' : '#1E293B',
          }} 
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

export default function App() {
  return (
    <SafeAreaProvider>
      <ThemeProvider>
        <AppContent />
      </ThemeProvider>
    </SafeAreaProvider>
  );
}
