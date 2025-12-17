import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Sidebar } from './components/Layout/Sidebar'
import { MainContent } from './components/Layout/MainContent'
import { Toaster } from './components/ui/toaster'
import { useAuthStore } from './stores/authStore'
import { LoginDialog } from './components/Layout/LoginDialog'

export type TabType = 'generate' | 'gallery' | 'history' | 'queue' | 'workflow' | 'settings'

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('generate')
  const { isAuthenticated, checkAuth } = useAuthStore()
  const { i18n } = useTranslation()

  useEffect(() => {
    // Check authentication on mount
    checkAuth()
    
    // Set theme
    document.documentElement.classList.add('dark')
  }, [checkAuth])

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-background">
      {!isAuthenticated ? (
        <LoginDialog />
      ) : (
        <>
          <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />
          <MainContent activeTab={activeTab} />
        </>
      )}
      <Toaster />
    </div>
  )
}

export default App
