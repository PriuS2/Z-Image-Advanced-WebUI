import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import { authApi } from '../api/auth'

interface User {
  id: number
  username: string
  settings: Record<string, unknown>
}

interface AuthState {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  
  login: (username: string, password: string) => Promise<boolean>
  register: (username: string, password: string) => Promise<boolean>
  logout: () => void
  checkAuth: () => Promise<void>
  clearError: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,

      login: async (username: string, password: string) => {
        set({ isLoading: true, error: null })
        try {
          const response = await authApi.login(username, password)
          set({
            token: response.access_token,
            isAuthenticated: true,
            isLoading: false,
          })
          
          // Fetch user info
          const user = await authApi.getMe(response.access_token)
          set({ user })
          
          return true
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Login failed',
            isLoading: false,
          })
          return false
        }
      },

      register: async (username: string, password: string) => {
        set({ isLoading: true, error: null })
        try {
          await authApi.register(username, password)
          // Auto login after registration
          return get().login(username, password)
        } catch (error) {
          set({
            error: error instanceof Error ? error.message : 'Registration failed',
            isLoading: false,
          })
          return false
        }
      },

      logout: () => {
        set({
          user: null,
          token: null,
          isAuthenticated: false,
          error: null,
        })
      },

      checkAuth: async () => {
        const token = get().token
        if (!token) {
          set({ isAuthenticated: false })
          return
        }

        try {
          const user = await authApi.getMe(token)
          set({ user, isAuthenticated: true })
        } catch {
          set({
            user: null,
            token: null,
            isAuthenticated: false,
          })
        }
      },

      clearError: () => set({ error: null }),
    }),
    {
      name: 'z-image-auth',
      partialize: (state) => ({ token: state.token }),
    }
  )
)
