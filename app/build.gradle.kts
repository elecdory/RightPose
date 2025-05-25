plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.rightpose"
    compileSdk = 33

    defaultConfig {
        applicationId = "com.example.rightpose"
        minSdk = 26   // 최소 SDK 버전 26
        targetSdk = 33
        versionCode = 1
        versionName = "1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }

    // 필요한 경우 추가 Compose 설정 작성
}

val cameraxVersion = "1.2.2"

dependencies {
    // Compose UI
    implementation("androidx.compose.ui:ui:1.3.3")
    implementation("androidx.compose.material:material:1.3.1")
    implementation("androidx.compose.ui:ui-tooling-preview:1.3.3")
    debugImplementation("androidx.compose.ui:ui-tooling:1.3.3")

    // Material3 & Material Components
    implementation("androidx.compose.material3:material3:1.0.1")
    implementation("com.google.android.material:material:1.9.0")

    implementation("androidx.core:core-ktx:1.10.1")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // CameraX (PreviewView 사용 시 camera-view 의존성이 필요)
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")


    // ML Kit
    // 기존의 정확도 높은 라이브러리 대신 기본 포즈 감지 라이브러리 사용
    implementation("com.google.mlkit:pose-detection-accurate:18.0.0-beta3")
    implementation("com.google.mlkit:face-detection:16.1.5")
    // (정확도 높은 라이브러리는 제거)

    // Unit Test 용 JUnit 의존성
    testImplementation("junit:junit:4.13.2")

    // Instrumentation Test 용 의존성
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")

}

// 최신 DSL 사용: compilerOptions DSL (자세한 내용: https://kotl.in/u1r8ln)
tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
    kotlinOptions.jvmTarget = "1.8"
}
