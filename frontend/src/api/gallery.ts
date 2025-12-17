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
}
