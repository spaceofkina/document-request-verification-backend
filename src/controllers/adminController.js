const DocumentRequest = require('../models/DocumentRequest');
const User = require('../models/User');
const templateService = require('../services/templateService');
const moment = require('moment');

exports.getAllRequests = async (req, res) => {
    try {
        const {
            status,
            documentType,
            startDate,
            endDate,
            page = 1,
            limit = 10,
            sortBy = 'dateRequested',
            sortOrder = 'desc'
        } = req.query;
        
        let query = {};
        
        // Apply filters
        if (status) query.status = status;
        if (documentType) query.documentType = documentType;
        if (startDate || endDate) {
            query.dateRequested = {};
            if (startDate) query.dateRequested.$gte = new Date(startDate);
            if (endDate) query.dateRequested.$lte = new Date(endDate);
        }
        
        // Sort
        const sort = {};
        sort[sortBy] = sortOrder === 'desc' ? -1 : 1;
        
        // Pagination
        const skip = (parseInt(page) - 1) * parseInt(limit);
        
        const [requests, total] = await Promise.all([
            DocumentRequest.find(query)
                .populate('userId', 'firstName lastName email')
                .sort(sort)
                .skip(skip)
                .limit(parseInt(limit)),
            DocumentRequest.countDocuments(query)
        ]);
        
        res.json({
            success: true,
            data: requests,
            pagination: {
                page: parseInt(page),
                limit: parseInt(limit),
                total,
                pages: Math.ceil(total / parseInt(limit))
            }
        });
    } catch (error) {
        console.error('Get all requests error:', error);
        res.status(500).json({ error: 'Server error' });
    }
};

exports.getRequestDetails = async (req, res) => {
    try {
        const { id } = req.params;
        
        const request = await DocumentRequest.findById(id)
            .populate('userId', 'firstName lastName email contactNumber address')
            .populate('adminId', 'firstName lastName');
        
        if (!request) {
            return res.status(404).json({ error: 'Request not found' });
        }
        
        // Prepare response with secure file paths
        const requestData = request.toObject();
        
        // Convert file paths to downloadable URLs
        requestData.attachments = requestData.attachments.map(att => ({
            ...att,
            downloadUrl: `/api/admin/requests/${id}/attachments/${att.filename}`
        }));
        
        res.json({
            success: true,
            data: requestData
        });
    } catch (error) {
        console.error('Get request details error:', error);
        res.status(500).json({ error: 'Server error' });
    }
};

exports.updateRequestStatus = async (req, res) => {
    try {
        const { id } = req.params;
        const { status, adminNotes } = req.body;
        const adminId = req.user.id;
        
        // Validate status
        const validStatuses = [
            'verified', 'failed', 'approved', 
            'declined', 'ready-to-claim'
        ];
        
        if (!validStatuses.includes(status)) {
            return res.status(400).json({ 
                error: 'Invalid status' 
            });
        }
        
        const request = await DocumentRequest.findById(id);
        if (!request) {
            return res.status(404).json({ error: 'Request not found' });
        }
        
        // Update request
        request.status = status;
        request.adminNotes = adminNotes;
        request.adminId = adminId;
        request.dateProcessed = new Date();
        
        await request.save();
        
        res.json({
            success: true,
            message: `Request ${status} successfully`,
            data: request
        });
    } catch (error) {
        console.error('Update request status error:', error);
        res.status(500).json({ error: 'Server error' });
    }
};

exports.verifyDocument = async (req, res) => {
    try {
        const { id } = req.params;
        const { verificationStatus, notes } = req.body;
        const adminId = req.user.id;
        
        const request = await DocumentRequest.findById(id);
        if (!request) {
            return res.status(404).json({ error: 'Request not found' });
        }
        
        // Update verification
        request.verificationResult.isVerified = verificationStatus === 'verified';
        request.verificationResult.verificationDate = new Date();
        
        if (verificationStatus === 'verified') {
            request.status = 'verified';
        } else {
            request.status = 'failed';
            request.verificationResult.mismatchDetails = notes || 'Manual verification failed';
        }
        
        request.adminId = adminId;
        request.adminNotes = notes;
        
        await request.save();
        
        res.json({
            success: true,
            message: `Document ${verificationStatus}`,
            data: request
        });
    } catch (error) {
        console.error('Verify document error:', error);
        res.status(500).json({ error: 'Server error' });
    }
};

exports.getDashboardStats = async (req, res) => {
    try {
        const today = new Date();
        const startOfMonth = new Date(today.getFullYear(), today.getMonth(), 1);
        const startOfWeek = new Date(today);
        startOfWeek.setDate(today.getDate() - today.getDay());
        
        const [
            totalRequests,
            pendingRequests,
            approvedRequests,
            declinedRequests,
            monthlyRequests,
            weeklyRequests,
            documentTypeStats,
            recentRequests
        ] = await Promise.all([
            DocumentRequest.countDocuments(),
            DocumentRequest.countDocuments({ status: { $in: ['pending', 'under-review'] } }),
            DocumentRequest.countDocuments({ status: 'approved' }),
            DocumentRequest.countDocuments({ status: 'declined' }),
            DocumentRequest.countDocuments({ dateRequested: { $gte: startOfMonth } }),
            DocumentRequest.countDocuments({ dateRequested: { $gte: startOfWeek } }),
            DocumentRequest.aggregate([
                { $group: { _id: '$documentType', count: { $sum: 1 } } },
                { $sort: { count: -1 } }
            ]),
            DocumentRequest.find()
                .populate('userId', 'firstName lastName')
                .sort({ dateRequested: -1 })
                .limit(10)
        ]);
        
        res.json({
            success: true,
            data: {
                totalRequests,
                pendingRequests,
                approvedRequests,
                declinedRequests,
                monthlyRequests,
                weeklyRequests,
                documentTypeStats,
                recentRequests,
                timestamp: new Date()
            }
        });
    } catch (error) {
        console.error('Get dashboard stats error:', error);
        res.status(500).json({ error: 'Server error' });
    }
};

exports.generateDocumentPreview = async (req, res) => {
    try {
        const { id } = req.params;
        
        const request = await DocumentRequest.findById(id)
            .populate('userId', 'firstName lastName email contactNumber address');
        
        if (!request) {
            return res.status(404).json({ error: 'Request not found' });
        }
        
        // Generate document
        const pdfBuffer = await templateService.generateDocument(
            request.documentType,
            {
                ...request.toObject(),
                barangayName: req.user.barangayName || 'Barangay',
                userId: request.userId._id
            }
        );
        
        // Send as preview (inline)
        res.set({
            'Content-Type': 'application/pdf',
            'Content-Disposition': `inline; filename="preview_${request.requestId}.pdf"`,
            'Content-Length': pdfBuffer.length
        });
        
        res.send(pdfBuffer);
    } catch (error) {
        console.error('Generate preview error:', error);
        res.status(500).json({ error: 'Server error generating preview' });
    }
};

exports.downloadAttachment = async (req, res) => {
    try {
        const { id, filename } = req.params;
        
        const request = await DocumentRequest.findById(id);
        if (!request) {
            return res.status(404).json({ error: 'Request not found' });
        }
        
        const attachment = request.attachments.find(att => att.filename === filename);
        if (!attachment) {
            return res.status(404).json({ error: 'Attachment not found' });
        }
        
        const filePath = path.join(__dirname, '../../', attachment.path);
        
        // Check if file exists
        try {
            await fs.access(filePath);
        } catch (error) {
            return res.status(404).json({ error: 'File not found on server' });
        }
        
        res.download(filePath, attachment.filename);
    } catch (error) {
        console.error('Download attachment error:', error);
        res.status(500).json({ error: 'Server error' });
    }
};