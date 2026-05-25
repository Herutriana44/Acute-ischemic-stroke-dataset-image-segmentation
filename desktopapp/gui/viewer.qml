import QtQuick
import QtQuick3D
import QtQuick3D.Helpers
import QtQuick3D.AssetUtils

Rectangle {
    width: 1200
    height: 800
    color: "#202020"

    View3D {
        anchors.fill: parent

        environment: SceneEnvironment {
            backgroundMode: SceneEnvironment.Color
            clearColor: "#202020"
        }

        PerspectiveCamera {
            id: camera
            position: Qt.vector3d(0,0,150)

            clipNear: 0.1
            clipFar: 10000
        }

        DirectionalLight {
            eulerRotation.x: -45
            eulerRotation.y: -30
            brightness: 3
        }

        Node {
            id: orbitCenter
        }

        RuntimeLoader {
            id: loader

            source: Qt.resolvedUrl(
                "Plastinated_Human_Brain/Plastinated_Human_Brain.gltf"
            )
        }

        OrbitCameraController {
            camera: camera
            origin: orbitCenter
        }
    }
}