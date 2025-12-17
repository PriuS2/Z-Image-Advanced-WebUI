import { useTranslation } from 'react-i18next'
import { 
  Sparkles, 
  Image, 
  History, 
  ListTodo, 
  Workflow,
  Settings, 
  LogOut,
  Zap
} from 'lucide-react'
import { TabType } from '../../App'
import { useAuthStore } from '../../stores/authStore'

interface SidebarProps {
  activeTab: TabType
  onTabChange: (tab: TabType) => void
}

const tabs: { id: TabType; icon: React.ComponentType<{ className?: string }> }[] = [
  { id: 'generate', icon: Sparkles },
  { id: 'gallery', icon: Image },
  { id: 'history', icon: History },
  { id: 'queue', icon: ListTodo },
  { id: 'workflow', icon: Workflow },
  { id: 'settings', icon: Settings },
]

export function Sidebar({ activeTab, onTabChange }: SidebarProps) {
  const { t } = useTranslation()
  const { logout, user } = useAuthStore()

  return (
    <aside className="flex h-full w-16 flex-col items-center border-r border-border bg-card py-4 md:w-64">
      {/* Logo */}
      <div className="mb-8 flex items-center gap-2 px-4">
        <Zap className="h-8 w-8 text-primary" />
        <span className="hidden text-lg font-bold md:block">Z-Image</span>
      </div>

      {/* Navigation */}
      <nav className="flex flex-1 flex-col gap-2 px-2 md:px-4 w-full">
        {tabs.map((tab) => {
          const Icon = tab.icon
          const isActive = activeTab === tab.id
          
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`
                flex items-center gap-3 rounded-lg px-3 py-3 text-sm font-medium
                transition-colors duration-200
                ${isActive 
                  ? 'bg-primary text-primary-foreground' 
                  : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                }
              `}
            >
              <Icon className="h-5 w-5 flex-shrink-0" />
              <span className="hidden md:block">{t(`tabs.${tab.id}`)}</span>
            </button>
          )
        })}
      </nav>

      {/* User section */}
      <div className="mt-auto flex w-full flex-col gap-2 border-t border-border px-2 pt-4 md:px-4">
        {user && (
          <div className="hidden items-center gap-2 px-3 py-2 md:flex">
            <div className="h-8 w-8 rounded-full bg-primary/20 flex items-center justify-center">
              <span className="text-sm font-medium text-primary">
                {user.username[0].toUpperCase()}
              </span>
            </div>
            <span className="text-sm font-medium">{user.username}</span>
          </div>
        )}
        
        <button
          onClick={logout}
          className="flex items-center gap-3 rounded-lg px-3 py-3 text-sm font-medium
            text-muted-foreground hover:bg-destructive/10 hover:text-destructive
            transition-colors duration-200"
        >
          <LogOut className="h-5 w-5 flex-shrink-0" />
          <span className="hidden md:block">{t('auth.logout')}</span>
        </button>
      </div>
    </aside>
  )
}
