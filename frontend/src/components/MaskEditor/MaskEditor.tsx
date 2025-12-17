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
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const maskCanvasRef = useRef<HTMLCanvasElement>(null)
  
  const [isDrawing, setIsDrawing] = useState(false)
  const [brushSize, setBrushSize] = useState(30)
  const [tool, setTool] = useState<'brush' | 'eraser'>('brush')
  const [zoom, setZoom] = useState(1)

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
      canvas.width = img.width
      canvas.height = img.height
      maskCanvas.width = img.width
      maskCanvas.height = img.height
      
      ctx.drawImage(img, 0, 0)
      
      // Initialize mask as transparent
      maskCtx.fillStyle = 'rgba(0, 0, 0, 0)'
      maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height)
    }
    img.src = image
  }, [image])

  const draw = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return

    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return

    const maskCtx = maskCanvas.getContext('2d')
    if (!maskCtx) return

    const rect = maskCanvas.getBoundingClientRect()
    const scaleX = maskCanvas.width / rect.width
    const scaleY = maskCanvas.height / rect.height
    
    const x = (e.clientX - rect.left) * scaleX
    const y = (e.clientY - rect.top) * scaleY

    maskCtx.beginPath()
    maskCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2)
    
    if (tool === 'brush') {
      maskCtx.fillStyle = 'rgba(255, 255, 255, 1)'
      maskCtx.fill()
    } else {
      maskCtx.globalCompositeOperation = 'destination-out'
      maskCtx.fill()
      maskCtx.globalCompositeOperation = 'source-over'
    }

    // Notify parent of mask change
    if (onMaskChange) {
      onMaskChange(maskCanvas.toDataURL('image/png'))
    }
  }, [isDrawing, brushSize, tool, onMaskChange])

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    setIsDrawing(true)
    draw(e)
  }

  const handleMouseUp = () => {
    setIsDrawing(false)
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    draw(e)
  }

  const clearMask = () => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return

    const maskCtx = maskCanvas.getContext('2d')
    if (!maskCtx) return

    maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)

    if (onMaskChange) {
      onMaskChange(maskCanvas.toDataURL('image/png'))
    }
  }

  const downloadMask = () => {
    const maskCanvas = maskCanvasRef.current
    if (!maskCanvas) return

    const link = document.createElement('a')
    link.download = 'mask.png'
    link.href = maskCanvas.toDataURL('image/png')
    link.click()
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Toolbar */}
      <div className="flex items-center gap-2 flex-wrap">
        <div className="flex rounded-md border border-border overflow-hidden">
          <button
            onClick={() => setTool('brush')}
            className={`p-2 ${tool === 'brush' ? 'bg-primary text-primary-foreground' : 'hover:bg-accent'}`}
            title="Brush"
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
        className="relative overflow-auto rounded-lg border border-border bg-muted/20"
        style={{ maxHeight: '70vh' }}
      >
        <div style={{ transform: `scale(${zoom})`, transformOrigin: 'top left' }}>
          {/* Base image canvas */}
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0"
            style={{ maxWidth: '100%' }}
          />
          
          {/* Mask canvas (overlay) */}
          <canvas
            ref={maskCanvasRef}
            onMouseDown={handleMouseDown}
            onMouseUp={handleMouseUp}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseUp}
            className="relative cursor-crosshair"
            style={{ 
              maxWidth: '100%',
              opacity: 0.5,
              mixBlendMode: 'multiply',
            }}
          />
        </div>
      </div>

      {/* Brush preview */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <div 
          className="rounded-full border border-primary bg-primary/30"
          style={{ width: brushSize, height: brushSize }}
        />
        <span>Brush preview</span>
      </div>
    </div>
  )
}
