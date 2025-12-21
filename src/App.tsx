import { useRef, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";

/* ===============================
   CORE ANALYTICS (NAMED EXPORTS)
================================ */
import { CSVUploader } from "./components/CSVUploader";
import { AnalyticsDashboard } from "./components/AnalyticsDashboard";
import { AIInsightsPanel } from "./components/AIInsightsPanel";
import { ExecutiveSummary } from "./components/ExecutiveSummary";

/* ===============================
   LANDING SECTIONS (DEFAULT EXPORTS)
================================ */
import HeroSection from "./components/HeroSection";
import ProductStory from "./components/ProductStory";
import HowItWorks from "./components/HowItWorks";
import Capabilities from "./components/Capabilities";
import EthicsGovernance from "./components/EthicsGovernance";
import Footer from "./components/Footer";

const queryClient = new QueryClient();

const App = () => {
  const uploadRef = useRef<HTMLDivElement>(null);
  const [analysisData, setAnalysisData] = useState<any>(null);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />

        {/* HERO */}
        <HeroSection
          onUploadClick={() =>
            uploadRef.current?.scrollIntoView({
              behavior: "smooth",
              block: "start",
            })
          }
        />

        {/* MARKETING SECTIONS */}
        <ProductStory />
        <HowItWorks />
        <Capabilities />
        <EthicsGovernance />

        {/* CSV UPLOAD */}
        <CSVUploader uploadRef={uploadRef} onDataLoaded={setAnalysisData} />

        {/* RESULTS (ONLY AFTER UPLOAD) */}
        {analysisData && (
          <>
            <AnalyticsDashboard metrics={analysisData.metrics} />

            <AIInsightsPanel
              mlWeights={analysisData.ml_weights}
              rag={analysisData.rag_explanations}
              topRisks={analysisData.top_risks}
            />

            <ExecutiveSummary text={analysisData.executive_summary} />
          </>
        )}

        <Footer />
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
