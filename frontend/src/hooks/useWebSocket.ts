import { useEffect, useRef, useCallback } from 'react'
import { useAuthStore } from '../stores/authStore'
import { useGenerationStore } from '../stores/generationStore'

interface WebSocketMessage {
  type: string
  task_id?: number
  progress?: number
  current_step?: number
  total_steps?: number
  elapsed_time?: number
  estimated_remaining?: number
  current_node?: string
  image?: string
  image_path?: string
  error?: string
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)
  const { user } = useAuthStore()
  const { setProgress, setLastGeneratedImage, resetProgress } = useGenerationStore()

  const connect = useCallback(() => {
    if (!user) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/${user.id}`

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected')
    }

    ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data)
        handleMessage(message)
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err)
      }
    }

    ws.onclose = () => {
      console.log('WebSocket disconnected')
      // Reconnect after 3 seconds
      reconnectTimeoutRef.current = window.setTimeout(connect, 3000)
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
  }, [user])

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'generation_progress':
        setProgress({
          progress: message.progress || 0,
          currentStep: message.current_step || 0,
          totalSteps: message.total_steps || 0,
          elapsedTime: message.elapsed_time || 0,
          estimatedRemaining: message.estimated_remaining || 0,
          currentNode: message.current_node || '',
        })
        break

      case 'preview_image':
        if (message.image) {
          setLastGeneratedImage(`data:image/png;base64,${message.image}`)
        }
        break

      case 'generation_complete':
        setProgress({
          progress: 100,
          isGenerating: false,
        })
        if (message.image_path) {
          setLastGeneratedImage(message.image_path)
        }
        break

      case 'generation_error':
        setProgress({
          isGenerating: false,
        })
        console.error('Generation error:', message.error)
        break

      case 'pong':
        // Heartbeat response
        break

      default:
        console.log('Unknown message type:', message.type)
    }
  }, [setProgress, setLastGeneratedImage])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  const send = useCallback((data: object) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const cancelGeneration = useCallback((taskId: number) => {
    send({ type: 'cancel', task_id: taskId })
  }, [send])

  useEffect(() => {
    if (user) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [user, connect, disconnect])

  return {
    send,
    cancelGeneration,
    isConnected: wsRef.current?.readyState === WebSocket.OPEN,
  }
}
