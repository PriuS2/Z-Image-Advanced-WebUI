import { create } from 'zustand'
import { persist } from 'zustand/middleware'

interface Settings {
  theme: 'dark' | 'light'
  language: 'ko' | 'en'
  realtimePreview: boolean
  autoSaveWorkflow: boolean
  generationDefaults: {
    width: number
    height: number
    steps: number
    controlScale: number
  }
  llmPreferences: {
    provider: string
    autoTranslate: boolean
    autoEnhance: boolean
  }
}

interface SettingsState {
  settings: Settings
  setSettings: (settings: Partial<Settings>) => void
  setTheme: (theme: 'dark' | 'light') => void
  setLanguage: (language: 'ko' | 'en') => void
}

const defaultSettings: Settings = {
  theme: 'dark',
  language: 'ko',
  realtimePreview: true,
  autoSaveWorkflow: true,
  generationDefaults: {
    width: 1024,
    height: 1024,
    steps: 25,
    controlScale: 0.75,
  },
  llmPreferences: {
    provider: 'openai',
    autoTranslate: false,
    autoEnhance: false,
  },
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      settings: defaultSettings,

      setSettings: (newSettings) =>
        set((state) => ({
          settings: { ...state.settings, ...newSettings },
        })),

      setTheme: (theme) =>
        set((state) => {
          // Update document class
          if (theme === 'dark') {
            document.documentElement.classList.add('dark')
          } else {
            document.documentElement.classList.remove('dark')
          }
          return {
            settings: { ...state.settings, theme },
          }
        }),

      setLanguage: (language) =>
        set((state) => ({
          settings: { ...state.settings, language },
        })),
    }),
    {
      name: 'z-image-settings',
    }
  )
)
