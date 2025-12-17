import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { X, Clock, CheckCircle, XCircle, Loader2, ListTodo } from 'lucide-react'
import { generationApi } from '../../api/generation'
import { useToast } from '../../hooks/useToast'

interface Task {
  id: number
  status: string
  progress: number
  created_at: string
  result_path?: string
  error_message?: string
}

export function QueueTab() {
  const { t } = useTranslation()
  const { error: toastError } = useToast()
  
  const [tasks, setTasks] = useState<Task[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [filter, setFilter] = useState<string | null>(null)

  const loadTasks = async () => {
    try {
      const data = await generationApi.getQueue(filter || undefined)
      setTasks(data)
    } catch (err) {
      toastError(t('errors.networkError'), String(err))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadTasks()
    const interval = setInterval(loadTasks, 5000) // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [filter])

  const handleCancel = async (taskId: number) => {
    try {
      await generationApi.cancelTask(taskId)
      loadTasks()
    } catch (err) {
      toastError(t('errors.networkError'), String(err))
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-4 w-4 text-yellow-500" />
      case 'running':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'failed':
      case 'cancelled':
        return <XCircle className="h-4 w-4 text-red-500" />
      default:
        return null
    }
  }

  const statusFilters = [
    { value: null, label: 'All' },
    { value: 'pending', label: t('queue.pending') },
    { value: 'running', label: t('queue.running') },
    { value: 'completed', label: t('queue.completed') },
    { value: 'failed', label: t('queue.failed') },
  ]

  return (
    <div className="h-full">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-xl font-bold">{t('queue.title')}</h2>
        
        <div className="flex items-center gap-1">
          {statusFilters.map((f) => (
            <button
              key={f.value || 'all'}
              onClick={() => setFilter(f.value)}
              className={`rounded-md px-3 py-1.5 text-sm transition-colors
                ${filter === f.value
                  ? 'bg-primary text-primary-foreground'
                  : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
                }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="flex h-64 items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : tasks.length === 0 ? (
        <div className="flex h-64 flex-col items-center justify-center gap-4 text-muted-foreground">
          <ListTodo className="h-16 w-16" />
          <p>{t('queue.noTasks')}</p>
        </div>
      ) : (
        <div className="space-y-2">
          {tasks.map((task) => (
            <div
              key={task.id}
              className="rounded-lg border border-border bg-card p-4"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {getStatusIcon(task.status)}
                  <span className="font-medium">Task #{task.id}</span>
                  <span className={`rounded-full px-2 py-0.5 text-xs capitalize
                    ${task.status === 'completed' ? 'bg-green-500/20 text-green-500' : ''}
                    ${task.status === 'running' ? 'bg-blue-500/20 text-blue-500' : ''}
                    ${task.status === 'pending' ? 'bg-yellow-500/20 text-yellow-500' : ''}
                    ${task.status === 'failed' || task.status === 'cancelled' ? 'bg-red-500/20 text-red-500' : ''}
                  `}>
                    {task.status}
                  </span>
                </div>
                
                <div className="flex items-center gap-4">
                  {task.status === 'running' && (
                    <div className="flex items-center gap-2">
                      <div className="h-2 w-32 rounded-full bg-muted overflow-hidden">
                        <div 
                          className="h-full bg-primary transition-all"
                          style={{ width: `${task.progress}%` }}
                        />
                      </div>
                      <span className="text-sm text-muted-foreground">{task.progress}%</span>
                    </div>
                  )}
                  
                  {(task.status === 'pending' || task.status === 'running') && (
                    <button
                      onClick={() => handleCancel(task.id)}
                      className="rounded p-1.5 hover:bg-destructive/10 hover:text-destructive transition-colors"
                      title={t('queue.cancel')}
                    >
                      <X className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </div>
              
              <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
                <span>{new Date(task.created_at).toLocaleString()}</span>
                {task.error_message && (
                  <span className="text-red-500">{task.error_message}</span>
                )}
              </div>
              
              {task.result_path && (
                <div className="mt-2">
                  <img
                    src={task.result_path}
                    alt="Result"
                    className="h-20 w-20 rounded object-cover"
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
