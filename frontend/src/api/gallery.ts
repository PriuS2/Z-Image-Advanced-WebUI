import { apiClient, handleApiError } from './client'

interface ImageResponse {
  id: number
  path: string
  thumbnail_path?: string
  prompt?: string
  params?: Record<string, unknown>
  generation_info?: Record<string, unknown>
  created_at: string
  is_favorite: boolean
}

interface ImageListResponse {
  images: ImageResponse[]
  total: number
  page: number
  page_size: number
}

export const galleryApi = {
  getImages: async (
    page: number = 1,
    pageSize: number = 20,
    favoriteOnly: boolean = false
  ): Promise<ImageListResponse> => {
    try {
      const response = await apiClient.get<ImageListResponse>('/gallery/images', {
        params: {
          page,
          page_size: pageSize,
          favorite_only: favoriteOnly,
        },
      })
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  getImage: async (imageId: number): Promise<ImageResponse> => {
    try {
      const response = await apiClient.get<ImageResponse>(`/gallery/images/${imageId}`)
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  deleteImage: async (imageId: number): Promise<void> => {
    try {
      await apiClient.delete(`/gallery/images/${imageId}`)
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  toggleFavorite: async (imageId: number): Promise<{ is_favorite: boolean }> => {
    try {
      const response = await apiClient.post<{ is_favorite: boolean }>(
        `/gallery/images/${imageId}/favorite`
      )
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  downloadImage: async (imageId: number): Promise<void> => {
    try {
      const response = await apiClient.get(`/gallery/images/${imageId}/download`, {
        responseType: 'blob',
      })
      
      // Create download link
      const blob = new Blob([response.data])
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `image_${imageId}.png`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  regenerate: async (imageId: number): Promise<{ task_id: number }> => {
    try {
      const response = await apiClient.post<{ task_id: number }>(
        `/gallery/images/${imageId}/regenerate`
      )
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },

  loadSettings: async (imageId: number): Promise<Record<string, unknown>> => {
    try {
      const response = await apiClient.get<Record<string, unknown>>(
        `/gallery/images/${imageId}/settings`
      )
      return response.data
    } catch (error) {
      throw new Error(handleApiError(error))
    }
  },
}
