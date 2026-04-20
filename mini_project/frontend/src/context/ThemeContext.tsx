import React, { createContext, useContext, useState, useEffect } from 'react';
import { saveUserPreferences, getUserPreferences } from '../services/firebase';

interface ThemeContextType {
  isDarkMode: boolean;
  notificationsEnabled: boolean;
  toggleDarkMode: (val: boolean) => void;
  toggleNotifications: (val: boolean) => void;
  isLoading: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [isLoading, setIsLoading] = useState(true);

  // Load preferences from Firebase on mount
  useEffect(() => {
    getUserPreferences().then((prefs) => {
      if (prefs) {
        setIsDarkMode(prefs.isDarkMode ?? false);
        setNotificationsEnabled(prefs.notificationsEnabled ?? true);
      }
      setIsLoading(false);
    });
  }, []);

  const toggleDarkMode = (val: boolean) => {
    setIsDarkMode(val);
    saveUserPreferences({ isDarkMode: val, notificationsEnabled }).catch(() => {});
  };

  const toggleNotifications = (val: boolean) => {
    setNotificationsEnabled(val);
    saveUserPreferences({ isDarkMode, notificationsEnabled: val }).catch(() => {});
  };

  return (
    <ThemeContext.Provider value={{ isDarkMode, notificationsEnabled, toggleDarkMode, toggleNotifications, isLoading }}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
}
