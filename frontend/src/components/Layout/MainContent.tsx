import { TabType } from '../../App'
import { GenerateTab } from '../FlowEditor/GenerateTab'
import { GalleryTab } from '../Gallery/GalleryTab'
import { HistoryTab } from '../History/HistoryTab'
import { QueueTab } from '../Queue/QueueTab'
import { WorkflowTab } from '../Workflow/WorkflowTab'
import { SettingsTab } from '../Settings/SettingsTab'
import { ProgressBar } from './ProgressBar'

interface MainContentProps {
  activeTab: TabType
}

export function MainContent({ activeTab }: MainContentProps) {
  const renderTab = () => {
    switch (activeTab) {
      case 'generate':
        return <GenerateTab />
      case 'gallery':
        return <GalleryTab />
      case 'history':
        return <HistoryTab />
      case 'queue':
        return <QueueTab />
      case 'workflow':
        return <WorkflowTab />
      case 'settings':
        return <SettingsTab />
      default:
        return null
    }
  }

  return (
    <main className="flex flex-1 flex-col overflow-hidden">
      {/* Progress bar at top */}
      <ProgressBar />
      
      {/* Tab content */}
      <div className="flex-1 overflow-auto p-4 md:p-6">
        {renderTab()}
      </div>
    </main>
  )
}
