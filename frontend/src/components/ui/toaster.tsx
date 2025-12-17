import { useToast } from '../../hooks/useToast'

export function Toaster() {
  const { toasts, removeToast } = useToast()

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`
            rounded-lg border p-4 shadow-lg transition-all duration-300
            ${toast.type === 'success' ? 'border-green-500 bg-green-500/10 text-green-500' : ''}
            ${toast.type === 'error' ? 'border-red-500 bg-red-500/10 text-red-500' : ''}
            ${toast.type === 'warning' ? 'border-yellow-500 bg-yellow-500/10 text-yellow-500' : ''}
            ${toast.type === 'info' ? 'border-blue-500 bg-blue-500/10 text-blue-500' : ''}
          `}
          onClick={() => removeToast(toast.id)}
        >
          <div className="font-medium">{toast.title}</div>
          {toast.description && (
            <div className="text-sm opacity-80">{toast.description}</div>
          )}
        </div>
      ))}
    </div>
  )
}
