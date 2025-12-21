import { motion } from "framer-motion";
import { ArrowDown } from "lucide-react";

const FloatingOrb = ({
  size,
  color,
  top,
  left,
  delay = 0,
}: {
  size: number;
  color: string;
  top: string;
  left: string;
  delay?: number;
}) => (
  <motion.div
    className={`absolute rounded-full blur-3xl opacity-30 ${color}`}
    style={{
      width: size,
      height: size,
      top,
      left,
    }}
    animate={{
      y: [0, -30, 10, 0],
      x: [0, 15, -10, 0],
      scale: [1, 1.1, 0.95, 1],
    }}
    transition={{
      duration: 8,
      repeat: Infinity,
      delay,
      ease: "easeInOut",
    }}
  />
);

interface HeroSectionProps {
  onUploadClick: () => void;
}

const HeroSection = ({ onUploadClick }: HeroSectionProps) => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0 animated-bg" />

      {/* Floating Orbs */}
      <FloatingOrb
        size={600}
        color="bg-purple"
        top="-20%"
        left="-10%"
        delay={0}
      />
      <FloatingOrb size={400} color="bg-blue" top="40%" left="70%" delay={2} />
      <FloatingOrb size={500} color="bg-cyan" top="60%" left="10%" delay={4} />
      <FloatingOrb
        size={300}
        color="bg-purple"
        top="10%"
        left="80%"
        delay={1}
      />

      {/* Grid Overlay */}
      <div
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage: `linear-gradient(hsl(var(--muted-foreground) / 0.1) 1px, transparent 1px),
                           linear-gradient(90deg, hsl(var(--muted-foreground) / 0.1) 1px, transparent 1px)`,
          backgroundSize: "60px 60px",
        }}
      />

      {/* Radial Glow */}
      <div className="absolute inset-0 bg-gradient-radial from-purple/10 via-transparent to-transparent" />

      {/* Content */}
      <div className="relative z-10 container mx-auto px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          {/* Badge */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8"
          >
            <span className="w-2 h-2 rounded-full bg-cyan animate-pulse" />
            <span className="text-sm font-medium text-muted-foreground">
              Applied AI Trust Monitoring Platform
            </span>
          </motion.div>

          {/* Title */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.8 }}
            className="font-display text-6xl md:text-7xl lg:text-8xl font-bold mb-6 tracking-tight"
          >
            <span className="gradient-text">TrustScope</span>
          </motion.h1>

          {/* Tagline */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.8 }}
            className="text-xl md:text-2xl text-muted-foreground max-w-3xl mx-auto mb-12 leading-relaxed"
          >
            Visualizing, predicting, and explaining{" "}
            <span className="text-foreground font-medium">
              trust in AI decisions
            </span>
          </motion.p>

          {/* CTA */}
          <motion.button
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7, duration: 0.8 }}
            whileHover={{
              scale: 1.05,
              boxShadow: "0 0 60px hsl(262 83% 58% / 0.5)",
            }}
            whileTap={{ scale: 0.98 }}
            onClick={onUploadClick}
            className="relative group px-8 py-4 rounded-xl font-semibold text-lg overflow-hidden"
            style={{ background: "var(--gradient-primary)" }}
          >
            <span className="relative z-10 flex items-center gap-2">
              Upload Decision Data
              <motion.span
                animate={{ y: [0, 4, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <ArrowDown className="w-5 h-5" />
              </motion.span>
            </span>
            <div className="absolute inset-0 bg-gradient-to-r from-cyan via-blue to-purple opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
};

export default HeroSection;
