<template>
  <view class="container">
    <view class="header">
      <text class="title">钢筋计数</text>
    </view>
    <view class="content">
      <button type="primary" @click="chooseImage" class="btn">选择图片</button>
      <image v-if="imageUrl" :src="imageUrl" class="preview-image"></image>
      <view v-if="result" class="result">
        <text class="result-text">钢筋数量: {{ result.count }}</text>
      </view>
    </view>
  </view>
</template>

<script>
export default {
  data() {
    return {
      imageUrl: '',
      result: null
    }
  },
  methods: {
    chooseImage() {
      uni.chooseImage({
        count: 1,
        sizeType: ['original', 'compressed'],
        sourceType: ['album', 'camera'],
        success: (res) => {
          this.imageUrl = res.tempFilePaths[0]
          this.detectRebar()
        }
      })
    },
    detectRebar() {
      // 这里将实现 ONNX 模型推理
      uni.showLoading({ title: '检测中...' })
      
      // 模拟检测结果
      setTimeout(() => {
        this.result = {
          count: Math.floor(Math.random() * 50) + 10
        }
        uni.hideLoading()
      }, 2000)
    }
  }
}
</script>

<style scoped>
.container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: #f5f5f5;
}

.header {
  background-color: #007aff;
  color: white;
  padding: 20rpx;
  text-align: center;
}

.title {
  font-size: 36rpx;
  font-weight: bold;
}

.content {
  flex: 1;
  padding: 40rpx;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.btn {
  width: 80%;
  margin-bottom: 40rpx;
}

.preview-image {
  width: 100%;
  height: 500rpx;
  margin: 40rpx 0;
  border-radius: 10rpx;
}

.result {
  margin-top: 40rpx;
  padding: 20rpx;
  background-color: white;
  border-radius: 10rpx;
  box-shadow: 0 2rpx 10rpx rgba(0, 0, 0, 0.1);
}

.result-text {
  font-size: 32rpx;
  color: #333;
}
</style>
