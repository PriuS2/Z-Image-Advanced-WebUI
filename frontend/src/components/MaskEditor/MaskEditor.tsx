import { useRef, useState, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { Brush, Eraser, RotateCcw, Download, ZoomIn, ZoomOut } from 'lucide-react'

interface MaskEditorProps {
  image: string
  onMaskChange?: (maskDataUrl: string) => void
  width?: number
  height?: number
}

export function MaskEditor({ image, onMaskChange, width = 512, height = 512 }: MaskEditorProps) {
  const { t } = useTranslation()
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const maskCanvasRef = useRef<HTMLCanvasElement>(null)
  
  const [isDrawing, setIsDrawing] = useState(false)
  const [brushSize, setBrushSize] = useState(30)
  const [tool, setTool] = useState<'brush' | 'eraser'>('brush')
  const [zoom, setZoom] = useState(1)
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 })
  const [lastPos, setLastPos] = useState<{ x: number; y: number } | null>(null)

  // Initialize canvases
  useEffect(() => {
    const canvas = canvasRef.current
    const maskCanvas = maskCanvasRef.current
    if (!canvas || !maskCanvas) return

    const ctx = canvas.getContext('2d')
    const maskCtx = maskCanvas.getContext('2d')
    if (!ctx || !maskCtx) return

    // Load image
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      // Calculate display size (max 600px width or height, maintaining aspect ratio)
      const maxSize = 600
      let displayWidth = img.width
      let displayHeight = img.height
      
      if (img.width > img.height) {
        if (img.width > maxSize) {
          displayWidth = maxSize
          displayHeight = (img.height / img.width) * maxSize
        }
      } else {
        if (img.height > maxSize) {
          displayHeight = maxSize
          displayWidth = (img.width / img.height) * maxSize
        }
      }
      
      // Set canvas internal size to match image (for high quality)
      canvas.width = img.width
      canvas.height = img.height
      maskCanvas.width = img.width
      maskCanvas.height = img.height
      
      // Store display size for CSS
      setCanvasSize({ width: displayWidth, height: displayHeight })
      
      ctx.drawImage(img, 0, 0)
      
      // Initialize mask as transparent
      maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
    }
    img.src = image
  }, [image])

  // Convert mask to white-on-transparent for export
  const getMaskDataUrl = useCallback(() => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return null

    // Create a temporary canvas for conversion
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = maskCanvas.width
    tempCanvas.height = maskCanvas.height
    const tempCtx = tempCanvas.getContext('2d')
    if (!tempCtx) return null

    // Get the mask data
    const maskCtx = maskCanvas.getContext('2d')
    if (!maskCtx) return null
    
    const imageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height)
    const data = imageData.data

    // Convert red areas to white (where alpha > 0, set to white)
    for (let i = 0; i < data.length; i += 4) {
      if (data[i + 3] > 0) {
        // Has alpha - this is a masked area, make it white
        data[i] = 255     // R
        data[i + 1] = 255 // G
        data[i + 2] = 255 // B
        data[i + 3] = 255 // A
      }
    }

    tempCtx.putImageData(imageData, 0, 0)
    return tempCanvas.toDataURL('image/png')
  }, [])

  // Draw function that directly uses coordinates - draws red for visibility
  const drawAt = useCallback((x: number, y: number) => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return

    const maskCtx = maskCanvas.getContext('2d')
    if (!maskCtx) return

    maskCtx.beginPath()
    maskCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2)
    
    if (tool === 'brush') {
      // Draw red for visibility
      maskCtx.fillStyle = 'rgba(255, 80, 80, 0.7)'
      maskCtx.fill()
    } else {
      maskCtx.globalCompositeOperation = 'destination-out'
      maskCtx.fill()
      maskCtx.globalCompositeOperation = 'source-over'
    }
  }, [brushSize, tool])

  // Draw line between two points for smooth strokes
  const drawLine = useCallback((x1: number, y1: number, x2: number, y2: number) => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return

    const maskCtx = maskCanvas.getContext('2d')
    if (!maskCtx) return

    maskCtx.beginPath()
    maskCtx.moveTo(x1, y1)
    maskCtx.lineTo(x2, y2)
    maskCtx.lineWidth = brushSize
    maskCtx.lineCap = 'round'
    maskCtx.lineJoin = 'round'
    
    if (tool === 'brush') {
      // Draw red for visibility
      maskCtx.strokeStyle = 'rgba(255, 80, 80, 0.7)'
      maskCtx.stroke()
    } else {
      maskCtx.globalCompositeOperation = 'destination-out'
      maskCtx.strokeStyle = 'rgba(255, 80, 80, 1)'
      maskCtx.stroke()
      maskCtx.globalCompositeOperation = 'source-over'
    }
  }, [brushSize, tool])

  const getCanvasCoordinates = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return null

    const rect = maskCanvas.getBoundingClientRect()
    const scaleX = maskCanvas.width / rect.width
    const scaleY = maskCanvas.height / rect.height
    
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    }
  }, [])

  const notifyMaskChange = useCallback(() => {
    if (!onMaskChange) return
    const dataUrl = getMaskDataUrl()
    if (dataUrl) {
      onMaskChange(dataUrl)
    }
  }, [onMaskChange, getMaskDataUrl])

  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    e.stopPropagation()
    
    const coords = getCanvasCoordinates(e)
    if (!coords) return

    setIsDrawing(true)
    setLastPos(coords)
    drawAt(coords.x, coords.y)
    notifyMaskChange()
  }, [getCanvasCoordinates, drawAt, notifyMaskChange])

  const handleMouseUp = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDrawing(false)
    setLastPos(null)
  }, [])

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    e.stopPropagation()
    
    if (!isDrawing) return

    const coords = getCanvasCoordinates(e)
    if (!coords) return

    if (lastPos) {
      drawLine(lastPos.x, lastPos.y, coords.x, coords.y)
    } else {
      drawAt(coords.x, coords.y)
    }
    setLastPos(coords)
    notifyMaskChange()
  }, [isDrawing, lastPos, getCanvasCoordinates, drawLine, drawAt, notifyMaskChange])

  const handleMouseLeave = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDrawing(false)
    setLastPos(null)
  }, [])

  const clearMask = useCallback(() => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return

    const maskCtx = maskCanvas.getContext('2d')
    if (!maskCtx) return

    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
    notifyMaskChange()
  }, [notifyMaskChange])

  const downloadMask = useCallback(() => {
    const dataUrl = getMaskDataUrl()
    if (!dataUrl) return

    const link = document.createElement('a')
    link.download = 'mask.png'
    link.href = dataUrl
    link.click()
  }, [getMaskDataUrl])

  return (
    <div className="flex flex-col gap-4">
      {/* Toolbar */}
      <div className="flex items-center gap-2 flex-wrap">
        <div className="flex rounded-md border border-border overflow-hidden">
          <button
            onClick={() => setTool('brush')}
            className={`p-2 ${tool === 'brush' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}`}
            title="Brush (draw mask)"
          >
            <Brush className="h-4 w-4" />
          </button>
          <button
            onClick={() => setTool('eraser')}
            className={`p-2 ${tool === 'eraser' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}`}
            title="Eraser"
          >
            <Eraser className="h-4 w-4" />
          </button>
        </div>

        <div className="flex items-center gap-2">
          <label className="text-sm">Size:</label>
          <input
            type="range"
            min={5}
            max={100}
            value={brushSize}
            onChange={(e) => setBrushSize(parseInt(e.target.value))}
            className="w-24 accent-primary"
          />
          <span className="text-sm w-8">{brushSize}</span>
        </div>

        <div className="flex rounded-md border border-border overflow-hidden">
          <button
            onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}
            className="p-2 hover:bg-accent"
            title="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <span className="px-2 py-1 text-sm border-x border-border">{Math.round(zoom * 100)}%</span>
          <button
            onClick={() => setZoom(Math.min(3, zoom + 0.25))}
            className="p-2 hover:bg-accent"
            title="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
        </div>

        <button
          onClick={clearMask}
          className="p-2 rounded-md border border-border hover:bg-accent"
          title="Clear mask"
        >
          <RotateCcw className="h-4 w-4" />
        </button>

        <button
          onClick={downloadMask}
          className="p-2 rounded-md border border-border hover:bg-accent"
          title="Download mask"
        >
          <Download className="h-4 w-4" />
        </button>
      </div>

      {/* Canvas container */}
      <div 
        ref={containerRef}
        className="relative overflow-auto rounded-lg border border-border flex items-start justify-center"
        style={{ 
          maxHeight: '60vh',
          background: 'repeating-conic-gradient(#404040 0% 25%, #303030 0% 50%) 50% / 16px 16px',
        }}
      >
        <div 
          className="relative"
          style={{ 
            transform: `scale(${zoom})`, 
            transformOrigin: 'top left',
            width: canvasSize.width > 0 ? `${canvasSize.width}px` : 'auto',
            height: canvasSize.height > 0 ? `${canvasSize.height}px` : 'auto',
          }}
        >
          {/* Base image canvas */}
          <canvas
            ref={canvasRef}
            style={{ 
              display: 'block',
              width: canvasSize.width > 0 ? `${canvasSize.width}px` : 'auto',
              height: canvasSize.height > 0 ? `${canvasSize.height}px` : 'auto',
            }}
          />
          
          {/* Mask canvas (overlay) - draws red for visibility */}
          <canvas
            ref={maskCanvasRef}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            style={{ 
              position: 'absolute',
              top: 0,
              left: 0,
              cursor: 'crosshair',
              width: canvasSize.width > 0 ? `${canvasSize.width}px` : 'auto',
              height: canvasSize.height > 0 ? `${canvasSize.height}px` : 'auto',
            }}
          />
        </div>
      </div>

      {/* Instructions */}
      <div className="flex items-center gap-3 text-sm text-muted-foreground">
        <div className="flex items-center gap-2">
          <div 
            className="rounded-full bg-red-500/70"
            style={{ width: Math.min(brushSize, 40), height: Math.min(brushSize, 40) }}
          />
          <span>Brush ({brushSize}px)</span>
        </div>
        <span className="text-xs">• Red areas will be inpainted</span>
      </div>
    </div>
  )
}
