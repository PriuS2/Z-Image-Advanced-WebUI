import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Heart, Trash2, Download, RotateCcw, Settings, Image as ImageIcon, Loader2 } from 'lucide-react'
import { galleryApi } from '../../api/gallery'
import { useToast } from '../../hooks/useToast'

interface GalleryImage {
  id: number
  path: string
  thumbnail_path?: string
  prompt?: string
  params?: Record<string, unknown>
  generation_info?: Record<string, unknown>
  created_at: string
  is_favorite: boolean
}

export function GalleryTab() {
  const { t } = useTranslation()
  const { error: toastError } = useToast()
  
  const [images, setImages] = useState<GalleryImage[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedImage, setSelectedImage] = useState<GalleryImage | null>(null)
  const [page, setPage] = useState(1)
  const [total, setTotal] = useState(0)
  const [favoriteOnly, setFavoriteOnly] = useState(false)

  const loadImages = async () => {
    setIsLoading(true)
    try {
      const response = await galleryApi.getImages(page, 20, favoriteOnly)
      setImages(response.images)
      setTotal(response.total)
    } catch (err) {
      toastError(t('errors.networkError'), String(err))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    loadImages()
  }, [page, favoriteOnly])

  const handleToggleFavorite = async (imageId: number) => {
    try {
      const result = await galleryApi.toggleFavorite(imageId)
      setImages((prev) =>
        prev.map((img) =>
          img.id === imageId ? { ...img, is_favorite: result.is_favorite } : img
        )
      )
    } catch (err) {
      toastError(t('errors.networkError'), String(err))
    }
  }

  const handleDelete = async (imageId: number) => {
    try {
      await galleryApi.deleteImage(imageId)
      setImages((prev) => prev.filter((img) => img.id !== imageId))
      if (selectedImage?.id === imageId) {
        setSelectedImage(null)
      }
    } catch (err) {
      toastError(t('errors.networkError'), String(err))
    }
  }

  return (
    <div className="flex h-full gap-4">
      {/* Gallery grid */}
      <div className="flex-1">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-xl font-bold">{t('gallery.title')}</h2>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setFavoriteOnly(!favoriteOnly)}
              className={`flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors
                ${favoriteOnly 
                  ? 'bg-primary text-primary-foreground' 
                  : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'
                }`}
            >
              <Heart className={`h-4 w-4 ${favoriteOnly ? 'fill-current' : ''}`} />
              {t('gallery.favorite')}
            </button>
          </div>
        </div>

        {isLoading ? (
          <div className="flex h-64 items-center justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : images.length === 0 ? (
          <div className="flex h-64 flex-col items-center justify-center gap-4 text-muted-foreground">
            <ImageIcon className="h-16 w-16" />
            <p>{t('gallery.noImages')}</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
            {images.map((image) => (
              <div
                key={image.id}
                onClick={() => setSelectedImage(image)}
                className={`group relative cursor-pointer overflow-hidden rounded-lg border transition-all
                  ${selectedImage?.id === image.id 
                    ? 'border-primary ring-2 ring-primary' 
                    : 'border-border hover:border-primary/50'
                  }`}
              >
                <img
                  src={image.thumbnail_path || image.path}
                  alt={image.prompt || 'Generated image'}
                  className="aspect-square w-full object-cover"
                />
                
                {/* Overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent
                  opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleToggleFavorite(image.id)
                      }}
                      className="rounded-full bg-black/50 p-1.5 hover:bg-black/70 transition-colors"
                    >
                      <Heart className={`h-4 w-4 text-white ${image.is_favorite ? 'fill-red-500' : ''}`} />
                    </button>
                    
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        handleDelete(image.id)
                      }}
                      className="rounded-full bg-black/50 p-1.5 hover:bg-red-500/70 transition-colors"
                    >
                      <Trash2 className="h-4 w-4 text-white" />
                    </button>
                  </div>
                </div>
                
                {/* Favorite indicator */}
                {image.is_favorite && (
                  <div className="absolute right-2 top-2">
                    <Heart className="h-4 w-4 fill-red-500 text-red-500" />
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        
        {/* Pagination */}
        {total > 20 && (
          <div className="mt-4 flex justify-center gap-2">
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              className="rounded-md bg-secondary px-3 py-1 text-sm disabled:opacity-50"
            >
              Previous
            </button>
            <span className="px-3 py-1 text-sm">
              Page {page} of {Math.ceil(total / 20)}
            </span>
            <button
              onClick={() => setPage(page + 1)}
              disabled={page * 20 >= total}
              className="rounded-md bg-secondary px-3 py-1 text-sm disabled:opacity-50"
            >
              Next
            </button>
          </div>
        )}
      </div>

      {/* Detail panel */}
      {selectedImage && (
        <div className="w-80 flex-shrink-0 rounded-lg border border-border bg-card p-4">
          <img
            src={selectedImage.path}
            alt={selectedImage.prompt || 'Generated image'}
            className="mb-4 w-full rounded-lg"
          />
          
          <div className="mb-4 flex gap-2">
            <button className="flex-1 flex items-center justify-center gap-1 rounded-md bg-secondary py-2 text-sm hover:bg-secondary/80">
              <Download className="h-4 w-4" />
              {t('gallery.download')}
            </button>
            <button className="flex-1 flex items-center justify-center gap-1 rounded-md bg-secondary py-2 text-sm hover:bg-secondary/80">
              <RotateCcw className="h-4 w-4" />
              {t('gallery.regenerate')}
            </button>
          </div>
          
          <button className="w-full flex items-center justify-center gap-1 rounded-md bg-primary py-2 text-sm text-primary-foreground hover:bg-primary/90">
            <Settings className="h-4 w-4" />
            {t('gallery.loadSettings')}
          </button>
          
          {selectedImage.prompt && (
            <div className="mt-4">
              <h4 className="mb-1 text-sm font-medium">Prompt</h4>
              <p className="text-xs text-muted-foreground">{selectedImage.prompt}</p>
            </div>
          )}
          
          {selectedImage.generation_info && (
            <div className="mt-4">
              <h4 className="mb-1 text-sm font-medium">Generation Info</h4>
              <pre className="text-xs text-muted-foreground overflow-auto max-h-40">
                {JSON.stringify(selectedImage.generation_info, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
