// frontend/mobile/App.tsx
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { Camera } from 'react-native-vision-camera';
import { useQuery, gql } from '@apollo/client';
import { SafeAreaView, View, Text } from 'react-native';

const GET_DETECTIONS = gql`
  query GetDetections {
    detections {
      id
      type
      confidence
      timestamp
      location
    }
  }
`;

const LiveFeedScreen = () => {
  const [hasPermission, setHasPermission] = React.useState(false);
  const camera = React.useRef<Camera>(null);

  React.useEffect(() => {
    (async () => {
      const status = await Camera.requestCameraPermission();
      setHasPermission(status === 'authorized');
    })();
  }, []);

  const { loading, error, data } = useQuery(GET_DETECTIONS);

  if (!hasPermission) {
    return <Text>No camera permission</Text>;
  }

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <Camera
        ref={camera}
        style={{ flex: 1 }}
        device={devices.back}
        isActive={true}
        photo={true}
        video={true}
      />
      <DetectionOverlay detections={data?.detections} />
    </SafeAreaView>
  );
};

// frontend/mobile/components/DetectionOverlay.tsx
interface Detection {
  id: string;
  type: string;
  confidence: number;
  timestamp: string;
  location: string;
}

const DetectionOverlay: React.FC<{ detections: Detection[] }> = ({ detections }) => {
  return (
    <View style={styles.overlay}>
      {detections?.map((detection) => (
        <View key={detection.id} style={styles.detection}>
          <Text style={styles.detectionText}>
            {detection.type} - {Math.round(detection.confidence * 100)}%
          </Text>
        </View>
      ))}
    </View>
  );
};

// frontend/mobile/components/ARView.tsx
import { ViroARScene, ViroARSceneNavigator } from '@viro-community/react-viro';

const ARScene = () => {
  return (
    <ViroARScene>
      {/* AR content */}
    </ViroARScene>
  );
};

const ARView: React.FC = () => {
  return (
    <ViroARSceneNavigator
      initialScene={{
        scene: ARScene,
      }}
      style={{ flex: 1 }}
    />
  );
};

// frontend/mobile/services/api.ts
import { ApolloClient, InMemoryCache } from '@apollo/client';

export const client = new ApolloClient({
  uri: 'http://localhost:8000/graphql',
  cache: new InMemoryCache(),
});
